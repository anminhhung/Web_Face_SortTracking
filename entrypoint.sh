python3 manage.py db init
python3 manage.py db migrate -m "init db"
python3 manage.py db upgrade

python3 app.py