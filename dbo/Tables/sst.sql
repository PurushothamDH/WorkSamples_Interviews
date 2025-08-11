CREATE TABLE [dbo].[sst] (
    [RowID]        INT           NULL,
    [OrderID]      VARCHAR (20)  NULL,
    [OrderDate]    INT           NULL,
    [ShipDate]     INT           NULL,
    [ShipMode]     VARCHAR (14)  NULL,
    [CustomerID]   VARCHAR (11)  NULL,
    [CustomerName] VARCHAR (22)  NULL,
    [Segment]      VARCHAR (11)  NULL,
    [Country]      VARCHAR (13)  NULL,
    [City]         VARCHAR (17)  NULL,
    [State]        VARCHAR (20)  NULL,
    [PostalCode]   INT           NULL,
    [Region]       VARCHAR (7)   NULL,
    [ProductID]    VARCHAR (15)  NULL,
    [Category]     VARCHAR (15)  NULL,
    [SubCategory]  VARCHAR (12)  NULL,
    [ProductName]  VARCHAR (127) NULL,
    [Sales]        VARCHAR (500) NULL,
    [Quantity]     VARCHAR (500) NULL,
    [Discount]     VARCHAR (500) NULL,
    [Profit]       VARCHAR (500) NULL
);
GO

