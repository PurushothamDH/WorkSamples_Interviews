SELECT
    Order_ID,
    Customer_Name,
    Product_Name,
    Quantity,
    Sales,
    Profit
FROM {{ source('samplesuperstore', 'orders') }}
WHERE Sales IS NOT NULL
