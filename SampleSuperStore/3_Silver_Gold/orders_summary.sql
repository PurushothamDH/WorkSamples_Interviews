SELECT
    Customer_Name,
    SUM(Sales) AS Total_Sales,
    SUM(Profit) AS Total_Profit,
    COUNT(Order_ID) AS Order_Count
FROM {{ ref('orders_clean') }}
GROUP BY Customer_Name
