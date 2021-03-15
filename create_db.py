import datetime
from flask_sqlalchemy import SQLAlchemy
# from app import db
from app import db

class BaseModel(db.Model):
    """Base data model for all objects"""
    __abstract__ = True

    def __init__(self, *args):
        super().__init__(*args)

    def __repr__(self):
        """Define a base way to print models"""
        return '%s(%s)' % (self.__class__.__name__, {
            column: value
            for column, value in self._to_dict().items()
        })

    def json(self):
        """
                Define a base way to jsonify models, dealing with datetime objects
        """
        return {
            column: value if not isinstance(value, datetime.date) else value.strftime('%Y-%m-%d')
            for column, value in self._to_dict().items()
        }

class Person(BaseModel, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    time = db.Column(db.DateTime, index=True)
    image_path = db.Column(db.String(200), index=True)

    def __init__(self, time, image_path):
        self.time = time
        self.image_path = image_path
    
    def __repr__(self):
        image_path = '<Image_path: {}>\n>'.format(self.image_path)

        return 
