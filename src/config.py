from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / 'artifacts'; ARTIFACTS.mkdir(exist_ok=True)
INTENTS = ['device_technical','apple_id_account','icloud','app_store','billing_payment','subscription','purchase_order','repair_service','complaint','other']
