# inventory
wine inventory management

## 准备输入数据
从ERP中以csv格式导出："北京1.csv", "北京2.csv", "北京3.csv"，"杭州.csv"， "广州.csv"， "代发.csv"等仓库数据，固定删除文件和库存表
统一放在一个文件夹中，文件夹路径可以通过main函数中的input_dir变量设置。

## 运行程序
点击运行，各项检测通过, 没有报错, 直到提示信息: "等待新品分类, 完成后按回车继续..."时,
此时在程序运行目录下已生成成本价偏差文件(bias.xlsx)和新品文件(no_category.xlsx).
打开新品文件, 手动修改最后一列的分类信息, 完成后保存并关闭no_category.xlsx文件.

## 手动操作
切换到程序运行界面, 点击回车, 继续程序运行. 完成后在运行目录下生成新的库存表文件.

## 补充说明
1. 产品分类信息文件（category.csv）
分类文件记录库存表中的"畅饮", "字典", "闪购&酒局", "会员", "香槟", "小甜水", "周边", "日本酒&威士忌&烈酒", "啤酒", "包材"分类条目中的
产品信息（条形码,产品品名,成本价,分类）。库存表更新后再运行程序，产品信息会覆盖原有分类文件中的产品信息。
建议保留每次运行生成的分类文件。
生成分类文件category.csv时，默认自动删除第一列为“汇总”的条目。
2. 历史分类文件(category_history.csv)
目前无用.
