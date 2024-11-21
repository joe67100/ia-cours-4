import os
from picsellia import Client


def main():
    client = Client(
        api_token=os.getenv("PICSELLIA_USER_TOKEN"),
        organization_name="Picsalex-MLOps",
    )
    dataset = client.get_dataset_by_id("01930b78-95d2-7ec5-9e09-3e2b00ee4af4")
    print(dataset.name)


if __name__ == "__main__":
    main()
