import design_bench as db

db.make("GFP-ResNet-v0", oracle_kwargs=dict(fit=True))
db.make("TFBind8-ResNet-v0", oracle_kwargs=dict(fit=True))
db.make("ChEMBL-ResNet-v0", oracle_kwargs=dict(fit=True))
db.make("UTR-ResNet-v0", oracle_kwargs=dict(fit=True))
