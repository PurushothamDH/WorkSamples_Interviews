CREATE TABLE [dbo].[SampleSuperStore] (
    [RowId]        INT             NULL,
    [OrderId]      VARCHAR (20)    NULL,
    [OrderDate]    DATE            NULL,
    [ShipDate]     DATE            NULL,
    [ShipMode]     VARCHAR (14)    NULL,
    [CustomerId]   VARCHAR (11)    NULL,
    [CustomerName] VARCHAR (22)    NULL,
    [Segment]      VARCHAR (11)    NULL,
    [Country]      VARCHAR (13)    NULL,
    [City]         VARCHAR (17)    NULL,
    [State]        VARCHAR (20)    NULL,
    [PostalCode]   INT             NULL,
    [Region]       VARCHAR (7)     NULL,
    [ProductId]    VARCHAR (15)    NULL,
    [Category]     VARCHAR (15)    NULL,
    [SubCategory]  VARCHAR (12)    NULL,
    [ProductName]  VARCHAR (127)   NULL,
    [Sales]        DECIMAL (18, 2) NULL,
    [Quantity]     DECIMAL (18, 2) NULL,
    [Discount]     DECIMAL (18, 2) NULL,
    [Profit]       DECIMAL (18, 2) NULL
);
GO

