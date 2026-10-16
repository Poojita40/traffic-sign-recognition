import os
from utils.predictor import predict_traffic_sign

sample_dir = os.path.join('archive', 'traffic_Data', 'DATA', '14')
sample_file = os.path.join(sample_dir, os.listdir(sample_dir)[0])
print("Testing sample image:", sample_file)

ok, res, err = predict_traffic_sign(sample_file)
print("Prediction success:", ok)
if ok:
    best = res["best_prediction"]
    print(f"Predicted: {best['class_name']} (Class ID: {best['class_id']}) with {best['confidence_pct']}% confidence")
    print("Top 3 guesses:")
    for p in res["top_predictions"]:
        print(f"  Rank {p['rank']}: {p['class_name']} (ID {p['class_id']}) -> {p['confidence_pct']}%")
else:
    print("Error:", err)
