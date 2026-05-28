from flask import Flask
from app.controllers.home_controller import home_bp
from app.controllers.crisp_controller import crisp_bp
from app.controllers.business_controller import business_bp
from app.controllers.data_understanding_controller import data_understanding_bp
from app.controllers.data_engineering_controller import data_engineering_bp
from app.controllers.model_engineering_controller import model_engineering_bp
from app.controllers.model_evaluation_controller import model_evaluation_bp
from app.controllers.prediction_controller import prediction_bp

app = Flask(__name__, template_folder='app/templates', static_folder='app/static')

app.register_blueprint(home_bp)
app.register_blueprint(crisp_bp)
app.register_blueprint(business_bp)
app.register_blueprint(data_understanding_bp)
app.register_blueprint(data_engineering_bp)
app.register_blueprint(model_engineering_bp)
app.register_blueprint(model_evaluation_bp)
app.register_blueprint(prediction_bp)

if __name__ == '__main__':
    app.run(debug=True)
