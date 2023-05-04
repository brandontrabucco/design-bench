try:
    from .hopper_controller_oracle import HopperControllerOracle
except ImportError:
    pass

try:
    from .ant_morphology_oracle import AntMorphologyOracle
except ImportError:
    pass

try:
    from .dkitty_morphology_oracle import DKittyMorphologyOracle
except ImportError:
    pass

try:
    from .toy_continuous_oracle import ToyContinuousOracle
except ImportError:
    pass

try:
    from .nas_bench_oracle import NASBenchOracle
except ImportError:
    pass

try:
    from .tf_bind_8_oracle import TFBind8Oracle
except ImportError:
    pass

try:
    from .tf_bind_10_oracle import TFBind10Oracle
except ImportError:
    pass

try:
    from .toy_discrete_oracle import ToyDiscreteOracle
except ImportError:
    pass
