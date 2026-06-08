import base64
import json
import os
import re
from dataclasses import dataclass

import cv2
import requests

from module.logger import logger


@dataclass
class VisionRecoverResult:
    handled: bool
    scene: str = "unknown"
    action: str = "none"
    confidence: float = 0.0
    reason: str = ""


class VisionRecover:
    ACTIONS = {
        "click_center": (640, 360),
        "click_right_middle": (900, 360),
        "click_blank_right_middle": (900, 360),
        "click_blue_back": (55, 45),
        "click_confirm": (760, 460),
        "click_cancel": (530, 460),
        "click_prepare": (1180, 610),
        "click_battle_exit": (35, 35),
        "click_top_left": (35, 35),
        "click_reward": (640, 360),
        "recover_battle_result": (640, 360),
        "close_popup": (900, 360),
    }

    PASS_ACTIONS = {
        "none",
        "do_nothing_retry",
    }

    STOP_ACTIONS = {
        "raise_real_stuck",
        "raise_human_takeover",
        "unknown",
    }

    def __init__(self):
        self.api_key = os.environ.get("OAS_VISION_API_KEY", "").strip()
        self.base_url = os.environ.get("OAS_VISION_API_BASE", "https://api.sharesai.xyz/v1").rstrip("/")
        self.model = os.environ.get("OAS_VISION_MODEL", "codex").strip()
        self.timeout = float(os.environ.get("OAS_VISION_TIMEOUT", "20"))
        self.min_confidence = float(os.environ.get("OAS_VISION_MIN_CONFIDENCE", "0.55"))

    @property
    def enabled(self) -> bool:
        return bool(self.api_key and self.model)

    def recover(self, device, task_name: str, error_type: str, click_history=None, detect_record=None) -> VisionRecoverResult:
        if not self.enabled:
            logger.warning("Vision recover disabled: OAS_VISION_API_KEY or OAS_VISION_MODEL not set")
            return VisionRecoverResult(False, reason="disabled")

        image = getattr(device, "image", None)
        if image is None:
            logger.warning("Vision recover skipped: no screenshot image")
            return VisionRecoverResult(False, reason="no image")

        try:
            decision = self._ask_model(
                image=image,
                task_name=task_name,
                error_type=error_type,
                click_history=click_history or [],
                detect_record=detect_record or [],
            )
        except Exception as e:
            logger.warning(f"Vision recover request failed: {e}")
            return VisionRecoverResult(False, reason=str(e))

        scene = str(decision.get("scene", "unknown"))
        action = str(decision.get("action", "unknown"))
        reason = str(decision.get("reason", ""))
        try:
            confidence = float(decision.get("confidence", 0))
        except (TypeError, ValueError):
            confidence = 0.0

        logger.info(f"Vision recover decision: scene={scene}, action={action}, confidence={confidence:.2f}")
        if reason:
            logger.info(f"Vision recover reason: {reason[:160]}")

        if confidence < self.min_confidence:
            logger.warning("Vision recover confidence too low")
            return VisionRecoverResult(False, scene, action, confidence, reason)

        if action in self.STOP_ACTIONS:
            return VisionRecoverResult(False, scene, action, confidence, reason)

        if action in self.PASS_ACTIONS:
            device.stuck_record_clear()
            device.click_record_clear()
            return VisionRecoverResult(True, scene, action, confidence, reason)

        if action not in self.ACTIONS:
            logger.warning(f"Vision recover action not allowed: {action}")
            return VisionRecoverResult(False, scene, action, confidence, reason)

        x, y = self.ACTIONS[action]
        device.click(x, y, control_name=f"vision_recover_{action}")
        device.stuck_record_clear()
        device.click_record_clear()
        return VisionRecoverResult(True, scene, action, confidence, reason)

    def _ask_model(self, image, task_name: str, error_type: str, click_history, detect_record) -> dict:
        image_b64 = self._encode_image(image)
        allowed_actions = sorted(self.ACTIONS.keys() | self.PASS_ACTIONS | self.STOP_ACTIONS)
        prompt = (
            "你是阴阳师自动化脚本的异常恢复裁判。当前脚本触发了等待太久或点击过多。"
            "请只根据截图判断游戏是否真的卡死，还是有弹窗、结算页、准备页、奖励页、分享页、确认框等阻塞识别。"
            "你必须只返回一个 JSON 对象，不要输出解释性文字。"
            "字段: scene(string), confidence(number 0-1), action(string), reason(string)。"
            f"action 只能是这些值之一: {', '.join(allowed_actions)}。"
            "优先选择低风险动作。若看不懂或不确定，使用 raise_real_stuck。"
        )
        user_text = {
            "task_name": task_name,
            "error_type": error_type,
            "click_history": [str(x) for x in click_history][-15:],
            "detect_record": [str(x) for x in detect_record],
        }
        messages = [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": json.dumps(user_text, ensure_ascii=False)},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                    },
                ],
            },
        ]
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": 300,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            content = self._post_chat_completions(payload, headers)
        except requests.HTTPError as e:
            logger.warning(f"Vision recover chat/completions failed, try responses: {e}")
            content = self._post_responses(prompt, user_text, image_b64, headers)
        return self._parse_json(content)

    def _post_chat_completions(self, payload: dict, headers: dict) -> str:
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    def _post_responses(self, prompt: str, user_text: dict, image_b64: str, headers: dict) -> str:
        payload = {
            "model": self.model,
            "input": [
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": prompt}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": json.dumps(user_text, ensure_ascii=False)},
                        {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{image_b64}",
                        },
                    ],
                },
            ],
            "temperature": 0,
            "max_output_tokens": 300,
        }
        response = requests.post(
            f"{self.base_url}/responses",
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        if "output_text" in data:
            return data["output_text"]
        chunks = []
        for item in data.get("output", []):
            for content in item.get("content", []):
                text = content.get("text")
                if text:
                    chunks.append(text)
        return "\n".join(chunks)

    @staticmethod
    def _encode_image(image) -> str:
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        ok, buffer = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            raise ValueError("failed to encode screenshot")
        return base64.b64encode(buffer.tobytes()).decode("ascii")

    @staticmethod
    def _parse_json(content: str) -> dict:
        content = content.strip()
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", content, re.S)
            if not match:
                raise
            return json.loads(match.group(0))
