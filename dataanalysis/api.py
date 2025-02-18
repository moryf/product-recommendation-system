import flask as fk
import json
from analysis import recommend_products

app = fk.Flask(__name__)

# Route that returns the average spending per customer
@app.route('/avg-customer', methods=['GET'])
def average_spending_per_customer():
    with open('data/average_spending_per_customer.json') as f:
        data = json.load(f)
    return data
# Route that returns the best selling products
@app.route('/best-selling-products', methods=['GET'])
def best_selling_products():
    with open('data/best_selling_products.json') as f:
        data = json.load(f)
    return data

#Route that returns the best selling categories
@app.route('/best-selling-categories', methods=['GET'])
def best_selling_categories():
    with open('data/best_selling_categories.json') as f:
        data = json.load(f)
    return data

#Route that returns the cluster stats
@app.route('/cluster_stats', methods=['GET'])
def cluster_stats():
    with open('data/cluster_stats.json') as f:
        data = json.load(f)
    return data

#Route that takes customer id and returns recommended products
@app.route('/recommend-products/<customer_id>', methods=['GET'])
def recommend(customer_id):
    return recommend_products(customer_id)

if __name__ == '__main__':
    app.run(port=5000, debug=True)

