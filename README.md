## Prerequisites
Before running the app, you'll need to have the following installed on your system:
- Python 3
- pip3

## Deploy the Backend Service
- Clone this repository to your local machine.
    ```bash
    git clone git@github.com:ashwin-gangadhar-mdb/mdb-cflt-chatapp.git
    ```

- Install the project dependencies using pip
    ```
    pip3 install -r req.txt
    ```
- Update the MongoDB Connection String and OpenAI API KEY.
    - `vi .env.dev`
    - `mv .env.dev .env`

- Run the Python code.
    ```
    python3 backend/run.py
    ```
