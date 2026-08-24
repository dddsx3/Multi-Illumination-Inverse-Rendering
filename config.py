"""
配置文件
包含所有训练和模型的配置参数

Author: Python Engineer
Date: 2026-01-22
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import json
from pathlib import Path


@dataclass
class DataConfig:
    """数据相关配置"""

    root_dir: str = r'C:\Users\35702\Desktop\processed_data'
    train_scenes: List[str] = field(default_factory=list)
    val_scenes: List[str] = field(default_factory=list)
    test_scenes: List[str] = field(default_factory=list)

    num_lights: int = 5
    image_size: Tuple[int, int] = (256, 256)
    file_extension: str = '.png'

    batch_size: int = 4
    num_workers: int = 4  # 根据优化脚本的建议
    pin_memory: bool = True
    prefetch_factor: int = 4  # 根据优化脚本的建议
    persistent_workers: bool = True  # 训练时保持worker进程

    # Phase 1 (T1.4-A3)：旋转增强的 GT 同步（法线向量旋转）未实现，默认关闭；
    # 图像/GT 同步的裁剪与翻转已实现，可正常启用。
    max_rotation_angle: float = 0.0
    horizontal_flip_prob: float = 0.5

    def __post_init__(self):
        """验证配置"""
        if self.num_lights < 2:
            raise ValueError("num_lights 必须至少为 2")

        if self.batch_size < 1:
            raise ValueError("batch_size 必须至少为 1")

        if self.num_workers < 0:
            raise ValueError("num_workers 不能为负数")

        if self.prefetch_factor < 2:
            raise ValueError("prefetch_factor 必须至少为 2")
@dataclass
class ModelConfig:
    """模型相关配置"""

    num_images: int = 5
    base_channels: int = 32
    sh_order: int = 2

    use_edge_aware: bool = True
    use_directional_light: bool = False

    use_local_residual: bool = True
    residual_hidden_dim: int = 32

    def __post_init__(self):
        """验证配置"""
        if self.base_channels < 16:
            raise ValueError("base_channels 必须至少为 16")

        if self.sh_order not in [1, 2, 3]:
            raise ValueError("sh_order 必须为 1, 2 或 3")


@dataclass
class TrainConfig:
    """训练相关配置"""

    total_epochs: int = 100
    learning_rate: float = 5e-5
    weight_decay: float = 1e-6

    stage1_epochs: int = 30
    stage2_epochs: int = 30

    scheduler: str = 'cosine'
    step_size: int = 30
    gamma: float = 0.5
    min_lr: float = 1e-6

    use_amp: bool = False
    gradient_clip: Optional[float] = 1.0  # 添加梯度裁剪
    grad_monitor_interval: int = 10  # 梯度监控间隔
    grad_logging: bool = True  # 启用梯度日志
    weight_init: str = 'xavier'  # 权重初始化类型
    log_interval: int = 10
    tensorboard_interval: int = 50
    val_interval: int = 1
    vis_interval: int = 5
    save_interval: int = 10

    def __post_init__(self):
        """验证配置"""
        if self.learning_rate <= 0:
            raise ValueError("learning_rate 必须为正数")

        if self.total_epochs < 1:
            raise ValueError("total_epochs 必须至少为 1")

        if self.scheduler not in ['step', 'cosine', 'plateau', 'none']:
            raise ValueError("scheduler 必须为 'step', 'cosine', 'plateau' 或 'none'")

        if self.stage1_epochs + self.stage2_epochs >= self.total_epochs:
            raise ValueError("stage1_epochs + stage2_epochs 必须小于 total_epochs")


@dataclass
class LossConfig:
    """损失函数权重配置"""

    reconstruction: float = 1.0
    depth_smooth: float = 0.01
    albedo_smooth: float = 0.01
    shading_smooth_weight: float = 0.0  # 添加阴影平滑权重
    retinex_constraint_weight: float = 0.0
    weight_dist: float = 0.01
    weight_tv: float = 0.01
    sh_l2: float = 0.001
    sh_higher: float = 0.002
    residual_l1: float = 0.01
    residual_tv: float = 0.01
    sh_sparsity: float = 0.001
    albedo_consistency: float = 1.0
    def __post_init__(self):
        """验证配置"""
        if self.reconstruction <= 0:
            raise ValueError("reconstruction 权重必须为正数")

        if any(w < 0 for w in [
            self.depth_smooth, self.albedo_smooth, self.shading_smooth_weight,
            self.retinex_constraint_weight, self.weight_dist,
            self.weight_tv, self.sh_l2, self.sh_higher, self.sh_sparsity,
            self.residual_l1, self.residual_tv, self.albedo_consistency
        ]):
            raise ValueError("所有损失权重必须为非负数")


@dataclass
class PathConfig:
    """路径相关配置"""

    log_dir: str = '../logs'
    checkpoint_dir: str = '../checkpoints'
    vis_dir: str = '../visualizations'
    output_dir: str = '../outputs'

    def __post_init__(self):
        """创建必要的目录"""
        for path in [self.log_dir, self.checkpoint_dir, self.vis_dir, self.output_dir]:
            Path(path).mkdir(parents=True, exist_ok=True)


@dataclass
class Config:
    """完整配置"""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    paths: PathConfig = field(default_factory=PathConfig)

    seed: int = 42
    device: str = 'cuda'  # 确保这里设置为 'cuda'
    verbose: bool = True

    def __post_init__(self):
        """验证配置"""
        if self.seed < 0:
            raise ValueError("seed 必须为非负数")

        if self.device not in ['cuda', 'cpu']:
            raise ValueError("device 必须为 'cuda' 或 'cpu'")

    def update_from_dict(self, config_dict: dict):
        """从字典更新配置"""
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
            elif hasattr(self.data, key):
                setattr(self.data, key, value)
            elif hasattr(self.model, key):
                setattr(self.model, key, value)
            elif hasattr(self.train, key):
                setattr(self.train, key, value)
            elif hasattr(self.loss, key):
                setattr(self.loss, key, value)
            elif hasattr(self.paths, key):
                setattr(self.paths, key, value)

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'data': self.data.__dict__,
            'model': self.model.__dict__,
            'train': self.train.__dict__,
            'loss': self.loss.__dict__,
            'paths': self.paths.__dict__,
            'seed': self.seed,
            'device': self.device,
            'verbose': self.verbose
        }

    def save(self, path: str):
        """保存配置到JSON文件"""
        config_dict = self.to_dict()

        # 如果路径不是绝对路径，默认保存到 config_text 文件夹
        path_obj = Path(path)
        if not path_obj.is_absolute():
            # 创建 config_text 文件夹
            config_dir = Path('../config_text')
            config_dir.mkdir(parents=True, exist_ok=True)
            path_obj = config_dir / path_obj.name

        with open(path_obj, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        print(f"配置已保存到: {path_obj}")

    @classmethod
    def load(cls, path: str) -> 'Config':
        """从JSON文件加载配置"""
        path_obj = Path(path)

        # 如果路径不是绝对路径，尝试从 config_text 文件夹加载
        if not path_obj.is_absolute():
            config_dir = Path('../config_text')
            config_path = config_dir / path_obj.name

            # 如果 config_text 文件夹中存在该文件，则使用它
            if config_path.exists():
                path_obj = config_path
            # 否则使用原始路径

        with open(path_obj, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)

        config = cls()

        if 'data' in config_dict:
            config.data = DataConfig(**config_dict['data'])
        if 'model' in config_dict:
            config.model = ModelConfig(**config_dict['model'])
        if 'train' in config_dict:
            config.train = TrainConfig(**config_dict['train'])
        if 'loss' in config_dict:
            config.loss = LossConfig(**config_dict['loss'])
        if 'paths' in config_dict:
            config.paths = PathConfig(**config_dict['paths'])

        if 'seed' in config_dict:
            config.seed = config_dict['seed']
        if 'device' in config_dict:
            config.device = config_dict['device']
        if 'verbose' in config_dict:
            config.verbose = config_dict['verbose']

        print(f"配置已从 {path_obj} 加载")
        return config

    def print(self):
        """打印配置"""
        print("\n" + "="*80)
        print("配置信息")
        print("="*80)
        print(f"\n【数据配置】")
        for key, value in self.data.__dict__.items():
            print(f"  {key}: {value}")

        print(f"\n【模型配置】")
        for key, value in self.model.__dict__.items():
            print(f"  {key}: {value}")

        print(f"\n【训练配置】")
        for key, value in self.train.__dict__.items():
            print(f"  {key}: {value}")

        print(f"\n【损失配置】")
        for key, value in self.loss.__dict__.items():
            print(f"  {key}: {value}")

        print(f"\n【路径配置】")
        for key, value in self.paths.__dict__.items():
            print(f"  {key}: {value}")

        print(f"\n【其他配置】")
        print(f"  seed: {self.seed}")
        print(f"  device: {self.device}")
        print(f"  verbose: {self.verbose}")

        print("\n" + "="*80 + "\n")


def get_default_config() -> Config:
    """获取默认配置"""
    return Config()


def get_stage_loss_weights(stage: int) -> dict:
    """
    获取指定阶段的损失权重

    Args:
        stage: 训练阶段 (1, 2, 3)

    Returns:
        损失权重字典
    """
    stage_weights = {
        1: {
            'reconstruction': 1.0,
            'depth_smooth': 0.01,
            'albedo_smooth': 0.0,
            'weight_dist': 0.0,
            'weight_tv': 0.0,
            'sh_l2': 0.001,
            'sh_higher': 0.0,
            'sh_sparsity': 0.001,
            'albedo_consistency': 1.0,
            'residual_l1': 0.0,
            'residual_tv': 0.0,
            'shading_smooth_weight': 0.0,
            'retinex_constraint_weight': 0.0
        },
        2: {
            'reconstruction': 1.0,
            'depth_smooth': 0.01,
            'albedo_smooth': 0.01,
            'weight_dist': 0.01,
            'weight_tv': 0.01,
            'sh_l2': 0.001,
            'sh_higher': 0.002,
            'sh_sparsity': 0.001,
            'albedo_consistency': 1.0,
            'residual_l1': 0.0,
            'residual_tv': 0.0,
            'shading_smooth_weight': 0.0,
            'retinex_constraint_weight': 0.0
        },
        3: {
            'reconstruction': 1.0,
            'depth_smooth': 0.01,
            'albedo_smooth': 0.01,
            'weight_dist': 0.01,
            'weight_tv': 0.01,
            'sh_l2': 0.001,
            'sh_higher': 0.002,
            'sh_sparsity': 0.001,
            'albedo_consistency': 1.0,
            'residual_l1': 0.01,
            'residual_tv': 0.01,
            'shading_smooth_weight': 0.0,
            'retinex_constraint_weight': 0.0
        }
    }

    return stage_weights.get(stage, stage_weights[3])

AGGRESSIVE_ALBEDO_SMOOTH = {
    'albedo_smooth_weight': 2.5,
    'shading_smooth_weight': 0.2,
    'retinex_constraint_weight': 0.3,
    'reconstruction_weight': 0.8
}
if __name__ == "__main__":
    """测试配置模块"""
    print("="*80)
    print("配置模块测试")
    print("="*80)

    config = get_default_config()
    config.print()

    print("测试阶段损失权重:")
    for stage in [1, 2, 3]:
        weights = get_stage_loss_weights(stage)
        print(f"\n阶段 {stage}:")
        for key, value in weights.items():
            if value > 0:
                print(f"  {key}: {value}")

    print("\n保存配置...")
    config.save('config_test.json')

    print("\n加载配置...")
    loaded_config = Config.load('config_test.json')
    loaded_config.print()
