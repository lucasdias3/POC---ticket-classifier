import json
from pipeline import load_data, transform_data
from trainer import train_model
from inferencer import classify_ticket

def main():
    # Load and preprocess the data (load_data retorna lista de registros)
    data = load_data('data/knowledge_base.json')
    preprocessed_data = transform_data(data)

    # Train the classification model (train_model retorna tuple (model, X_test, y_test))
    model, _, _ = train_model(preprocessed_data)

    # Example of classifying a new ticket
    new_ticket = {
        "customer_segment": "Startup",
        "channel": "email",
        "text": "Estou tendo problemas com a conexão.",
    }
    classification = classify_ticket(model, new_ticket)
    print(f"Classification for the new ticket: {classification}")

if __name__ == "__main__":
    main()