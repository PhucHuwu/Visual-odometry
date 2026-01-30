"""Configuration Loader Utility

Load và merge YAML configuration files.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class ConfigLoader:
    """Load và quản lý configuration từ YAML files"""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize config loader.

        Args:
            config_path: Path tới config file. Nếu None, dùng default_config.yaml
        """
        if config_path is None:
            # Tìm config folder từ project root
            current = Path(__file__).resolve()
            project_root = current.parent.parent.parent
            config_path = project_root / "config" / "default_config.yaml"

        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = {}
        self._load_config()

    def _load_config(self) -> None:
        """Load config từ YAML file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file không tồn tại: {self.config_path}")

        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Lấy config value với nested key support.

        Args:
            key: Config key (hỗ trợ nested: "camera.calibration_file")
            default: Default value nếu key không tồn tại

        Returns:
            Config value hoặc default

        Example:
            >>> config.get("camera.calibration_file")
            "config/camera_params.yaml"
        """
        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any) -> None:
        """
        Set config value (runtime override, không save vào file).

        Args:
            key: Config key (hỗ trợ nested)
            value: Giá trị mới
        """
        keys = key.split('.')
        config = self.config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def load_additional(self, file_path: str) -> Dict[str, Any]:
        """
        Load thêm config file khác (ví dụ: algorithm_config, camera_params).

        Args:
            file_path: Path tới YAML file

        Returns:
            Loaded config dict
        """
        path = Path(file_path)

        # Nếu relative path, resolve từ project root
        if not path.is_absolute():
            current = Path(__file__).resolve()
            project_root = current.parent.parent.parent
            path = project_root / file_path

        if not path.exists():
            raise FileNotFoundError(f"Config file không tồn tại: {path}")

        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def __getitem__(self, key: str) -> Any:
        """Support dict-like access: config["key"]"""
        return self.get(key)

    def __setitem__(self, key: str, value: Any) -> None:
        """Support dict-like assignment: config["key"] = value"""
        self.set(key, value)

    def __repr__(self) -> str:
        return f"ConfigLoader(config_path='{self.config_path}')"
