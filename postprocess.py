import pandas as pd

df = pd.read_csv('test.csv')

final_results = []

for car_id in df['car_id'].unique():
    car_rows = df[df['car_id'] == car_id]

    # choose most common plate
    plate = car_rows['license_plate_text'].mode()[0]

    best_row = car_rows.loc[car_rows['text_score'].idxmax()]

    final_results.append({
        'car_id': car_id,
        'license_plate': plate,
        'confidence': best_row['text_score']
    })

final_df = pd.DataFrame(final_results)
final_df.to_csv('final_results.csv', index=False)

print(final_df)