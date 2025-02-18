import pandas as pd
import matplotlib.pyplot as plt

"""
    Using the pandas library to read a csv file containing the sample dataset
"""
df = pd.read_csv('data/sample_dataset.csv')

def read_csv(file_path):
    return pd.read_csv(file_path)

df.head()
#%%
"""
    Identifying the best selling products and categories
"""
best_selling_products = df['Product ID'].value_counts().head(5)
best_selling_categories = df['Product Category'].value_counts().head(5)

print("Best Selling Products")
print(best_selling_products)
print("\nBest Selling Categories")
print(best_selling_categories)


# Saving the best selling products and categories to a json file
best_selling_products.to_json('data/best_selling_products.json')
best_selling_categories.to_json('data/best_selling_categories.json')

#%%
"""
    Calculating the average spending per customer and saving it to a json file
"""

average_spending_per_customer = df.groupby('Customer ID')['Purchase Amount'].mean()
average_spending_per_customer = average_spending_per_customer.sort_values(ascending=False)
average_spending_per_customer.head(5)

# Saving the average spending per customer to a json file
average_spending_per_customer.to_json('data/average_spending_per_customer.json')

#%%
"""
    Plotting the best selling products and categories and saving the plots to png files
"""


# Plotting the best selling products
best_selling_products.plot(kind='bar', title='Best Selling Products')
plt.ylabel('Number of Sales')
plt.xlabel('Product ID')
plt.savefig('data/best_selling_products.png')
plt.show()

# Plotting the best selling categories
best_selling_categories.plot(kind='bar', title='Best Selling Categories')
plt.ylabel('Number of Sales')
plt.xlabel('Product Category')
plt.savefig('data/best_selling_categories.png')
plt.show()
#%%
"""
    Plotting average spending per customer distribution and saving the plot to a png file
"""

average_spending_per_customer.plot(kind='hist', title='Average Spending per Customer')
plt.ylabel('Number of Customers')
plt.xlabel('Average Spending')
plt.savefig('data/average_spending_per_customer.png')
plt.show()

#%%
"""
    Clustering customers based on spending and frequency of purchases
"""

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# Grouping data by customer ID and aggregating the total spending and frequency of purchases
customer_data = df.groupby('Customer ID').agg({
    'Purchase Amount': 'sum',  # Total spending
    'Customer ID': 'size'      # Frequency of purchases
}).rename(columns={'Customer ID': 'Purchase Count'}).reset_index()


# Standardizing the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(customer_data[['Purchase Amount', 'Purchase Count']])

# Clustering customers into 5 clusters
kmeans = KMeans(n_clusters=5, random_state=42)
customer_data['Cluster'] = kmeans.fit_predict(scaled_data)

#%%
"""
    Displaying the cluster labels and cluster names for each customer
"""

# Mapping cluster labels to cluster names
cluster_labels = {
    0: 'Frequent Buyers',
    1: 'Average Buyers',
    2: 'Window Shoppers',
    3: 'Biggest Spenders',
    4: 'Less Frequent Buyers'
}

customer_data['Cluster Name'] = customer_data['Cluster'].map(cluster_labels)

print(customer_data[['Customer ID', 'Purchase Amount', 'Purchase Count', 'Cluster', 'Cluster Name']])

#%%
"""
    Calculating cluster statistics and saving the results to a json file
    Cluster statistics include average spending, average purchase count, popular category and cluster name
"""

# Calculating cluster statistics
cluster_stats = customer_data.groupby('Cluster').agg({
    'Purchase Amount': 'mean',
    'Purchase Count': 'mean'
}).rename(columns={
    'Purchase Amount': 'Average Spending',
    'Purchase Count': 'Average Purchase Count'
})

# Calculating the most popular category for each cluster
cluster_categories = df.merge(customer_data[['Customer ID', 'Cluster']], on='Customer ID')
cluster_categories = cluster_categories.groupby('Cluster')['Product Category'].apply(lambda x: x.value_counts().idxmax())
cluster_stats['Popular Category'] = cluster_categories

# Adding cluster names to the cluster statistics
cluster_stats['Cluster Name'] = cluster_stats.index.map(cluster_labels)

print(cluster_stats)

# Saving the cluster statistics to a json file
cluster_stats.to_json('data/cluster_stats.json')

#%%
"""
    Generating a plot showing the distribution of customers across clusters
"""

plt.scatter(customer_data['Purchase Count'], customer_data['Purchase Amount'], c=customer_data['Cluster'])
plt.xlabel('Number of Purchases')
plt.ylabel('Total Spending')
plt.title('Customer Clustering')
plt.savefig('data/customer_clustering.png')
plt.show()

#%%
"""
    Collaborative Filtering Implementation - Recommending products to customers based on their purchase history
    
"""

# Importing required libraries
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# Creating a user-item matrix
# Each row represents a customer and each column represents a product
# The values represent the purchase amount
user_item_matrix = df.pivot_table(index='Customer ID', columns='Product ID', values='Purchase Amount', fill_value=0)

# Calculating cosine similarity between customers and products
# Customer similarity is calculated based on the purchase history of customers
customer_similarity = cosine_similarity(user_item_matrix)
product_similarity = cosine_similarity(user_item_matrix.T)

# Normalizing the similarity scores
scaler = MinMaxScaler()
customer_similarity = scaler.fit_transform(customer_similarity)

# Function to recommend products to a customer based on their purchase history
# The function returns the top N recommended products for the customer
def recommend_products(customer_id, top_n=5):
    # Get the index of the customer in the user-item matrix
    customer_idx = user_item_matrix.index.get_loc(customer_id)

    # Find the top N similar customers
    similar_customers = customer_similarity[customer_idx]
    similar_customers_idx = similar_customers.argsort()[-2:-top_n-2:-1]  # Exclude self
    # Sum the purchase history of similar customers to recommend products
    recommended_products = user_item_matrix.iloc[similar_customers_idx].sum(axis=0)

    # Exclude products that the customer has already purchased
    purchased_products = user_item_matrix.loc[customer_id] > 0
    recommended_products = recommended_products[~purchased_products]

    # Return the top N recommended products
    return recommended_products.sort_values(ascending=False).head(top_n).index.tolist()

#%%
# Example: Recommend products for customer 'C123'
customer_id = 'C123'
recommended_products = recommend_products(customer_id)
print(f"Recommended products for Customer {customer_id}: {recommended_products}")
#%%
"""
    Generating a PDF report containing the analysis results
"""
from reportlab.pdfgen import canvas

# Create a PDF report
pdf_filename = 'data/analysis_report.pdf'
c = canvas.Canvas(pdf_filename)
c.setPageSize((600,2000))
c.setFontSize(20)
c.drawString(100,1950, "Analysis Report")
c.setFontSize(16)
c.drawString(100,1900, "Best Selling Products")
c.drawInlineImage('data/best_selling_products.png', 100, 1575, width=400, height=300)
c.drawString(100,1550, "Best Selling Categories")
c.drawInlineImage('data/best_selling_categories.png', 100, 1225, width=400, height=300)
c.drawString(100,1200, "Average Spending per Customer")
c.drawInlineImage('data/average_spending_per_customer.png',100,875, width=400, height=300)
c.drawString(100,850, "Customer Clustering")
c.drawInlineImage('data/customer_clustering.png', 100, 525, width=400, height=300)
c.drawString(100,500, "A Example of Recommended Products")
c.drawString(100,450, "Recommended Products for Customer C123")
c.drawString(100,400, str(recommended_products))
c.save()
#%%
