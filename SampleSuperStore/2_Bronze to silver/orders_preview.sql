SELECT *
FROM {{ source('samplesuperstore', 'orders') }}
LIMIT 10
