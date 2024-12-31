import json
import csv
import random
import os
from faker import Faker

def generate_name_number_json(num_entries=500):
    fake = Faker()
    data = {}
    for _ in range(num_entries):
        name = f"{fake.first_name()} {fake.last_name()}"
        number = ''.join([str(random.randint(0, 9)) for _ in range(10)])
        data[name] = number
    return data

def generate_csv_data(json_data):
    csv_data = []
    for name, number in json_data.items():
        input_text = f"What is the 10-digit number of {name}?"
        rejected_completion = ''.join([str(random.randint(0, 9)) for _ in range(10)])
        while rejected_completion == number:
            rejected_completion = ''.join([str(random.randint(0, 9)) for _ in range(10)])
        accepted_completion = number
        csv_data.append([input_text, rejected_completion, accepted_completion])
    return csv_data

def save_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

def save_csv(data, filename):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Input_Text', 'Rejected_Completion', 'Accepted_Completion'])
        writer.writerows(data)

def main():
    # Create output directory if it doesn't exist
    output_dir = './data/name_number_query'
    os.makedirs(output_dir, exist_ok=True)

    # Generate and save JSON data
    json_data = generate_name_number_json()
    json_filename = os.path.join(output_dir, 'name_number_pairs.json')
    save_json(json_data, json_filename)
    print(f"JSON file saved: {json_filename}")

    # Generate and save CSV data
    csv_data = generate_csv_data(json_data)
    csv_filename = os.path.join(output_dir, 'name_number_query.csv')
    save_csv(csv_data, csv_filename)
    print(f"CSV file saved: {csv_filename}")

if __name__ == "__main__":
    main()

