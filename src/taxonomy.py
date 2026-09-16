from .config import INTENTS

def infer_intent(text: str) -> str:
    x=text.lower()
    rules=[
      ('billing_payment',['charged','charge','billing','payment','card','refund','purchase','money']),
      ('apple_id_account',['apple id','password','sign in','login','logged in','account','verification','locked','activation lock']),
      ('icloud',['icloud','backup','sync','storage']),
      ('app_store',['app store','itunes','download app','update app','in-app','application']),
      ('subscription',['subscription','subscribe','renewal','cancel subscription','apple music','apple tv','icloud+']),
      ('repair_service',['repair','service','genius','appointment','replacement','broken screen','service center']),
      ('purchase_order',['order','delivery','delivered','shipping','shipment','package','purchase','buy']),
      ('device_technical',['iphone','ipad','macbook','mac','airpods','apple watch','ios','ipados','macos','device','not working','error','crash','update','battery','wifi','bluetooth']),
      ('complaint',['complaint','terrible','awful','useless','angry','frustrated','disappointed','worst','hate']),
    ]
    for label,words in rules:
      if any(w in x for w in words): return label
    return 'other'
