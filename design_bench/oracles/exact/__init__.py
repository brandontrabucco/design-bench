try:
    from .ant_morphology_oracle import AntMorphologyOracle
except ImportError as e:
    print("Skipping AntMorphologyOracle import:", e)

try:
    from .cifar_nas_oracle import CIFARNASOracle
except ImportError as e:
    print("Skipping CIFARNASOracle import:", e)

try:
    from .dkitty_morphology_oracle import DKittyMorphologyOracle
except ImportError as e:
    print("Skipping DKittyMorphologyOracle import:", e)

try:
    from .hopper_controller_oracle import HopperControllerOracle
except ImportError as e:
    print("Skipping HopperControllerOracle import:", e)

try:
    from .hopper_controller_stochastic_oracle import HopperControllerStochasticOracle
except ImportError as e:
    print("Skipping HopperControllerStochasticOracle import:", e)

try:
    from .nas_bench_oracle import NASBenchOracle
except ImportError as e:
    print("Skipping NASBenchOracle import:", e)

try:
    from .tf_bind_8_oracle import TFBind8Oracle
except ImportError as e:
    print("Skipping TFBind8Oracle import:", e)

try:
    from .tf_bind_10_oracle import TFBind10Oracle
except ImportError as e:
    print("Skipping TFBind10Oracle import:", e)

try:
    from .toy_continuous_oracle import ToyContinuousOracle
except ImportError as e:
    print("Skipping ToyContinuousOracle import:", e)

try:
    from .toy_discrete_oracle import ToyDiscreteOracle
except ImportError as e:
    print("Skipping ToyDiscreteOracle import:", e)
