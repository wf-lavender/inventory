from bcolors import Bcolors
from datetime import datetime
import pandas as pd
import numpy as np
import os


ERROR_LEVEL = {
    "warn": ["警告", "Warnings", Bcolors.CYELLOW2],
    "fatal": ["错误", "Fatal", Bcolors.CRED2],
}


def check_bias(df1, df2, column, return_columns, threshold=0.5, suffixes=("_ERP", "_已确认")):
    merged_df = pd.merge(df1, df2, on="条形码", suffixes=("", suffixes[1]))
    merged_df.rename(columns={column: column+suffixes[0]}, inplace=True)
    suspect_df = merged_df.loc[(merged_df[column+suffixes[0]] == 0) | (merged_df[column+suffixes[1]] == 0) |
                               ((merged_df[column+suffixes[0]] - merged_df[column+suffixes[1]]).abs() >= threshold), :]
    # print(suspect_df[return_columns])
    return suspect_df[return_columns]


def check_missing(df_data, col_names, level="warn"):
    """
    :param df_data: <pandas.DataFrame>
    :param col_names: <str list or str>
    :param level: <str>
    :return:
    """
    suspect_lines = df_data.loc[df_data[col_names].isna().any(axis=1), :]
    if not suspect_lines.empty:
        print("{0} {2}: {1} 字段缺失 {0}".format(10*"*", str(col_names), ERROR_LEVEL[level][0]))
        print(suspect_lines)
        print(30*"*")
        return "{0}{2} {3}{1}".format(ERROR_LEVEL[level][2], Bcolors.CEND,
                                      len(suspect_lines), ERROR_LEVEL[level][1])
    else:
        return "{0} passed {1}".format(Bcolors.CGREEN2, Bcolors.CEND)


def check_duplicate(df_data, col_names, level="fatal"):
    """
    :param df_data: <pandas.DataFrame>
    :param col_names: <str list or str>
    :param level: <str>
    :return:
    """
    suspect_lines = df_data[df_data.duplicated(subset=col_names, keep=False)]
    if not suspect_lines.empty:
        print("{0} {2}: {1} 字段重复 {0}".format(10 * "*", str(col_names), ERROR_LEVEL[level][0]))
        print(suspect_lines.sort_values(by=col_names))
        print(50 * "*")
        return "{0} {2} {1}".format(ERROR_LEVEL[level][2], Bcolors.CEND, ERROR_LEVEL[level][1])
    else:
        return "{0} passed {1}".format(Bcolors.CGREEN2, Bcolors.CEND)


def check_value_range(df_data, column, choices, level="fatal"):
    """
    :param df_data: <pandas.DataFrame>
    :param column: <str>
    :param choices: <list>
    :param level: <str>
    :return:
    """
    wrong_data = df_data.loc[~df_data[column].isin(choices), :]
    if not wrong_data.empty:
        print("{0} {2}: {1} 设置错误 {0}".format(10 * "*", column, ERROR_LEVEL[level][0]))
        print(wrong_data)
        print("正确{0}范围: {1}".format(column, str(choices)))
        print(50 * "*")
        return "{0} {2} {1}".format(ERROR_LEVEL[level][2], Bcolors.CEND, ERROR_LEVEL[level][1])
    else:
        return "{0} passed {1}".format(Bcolors.CGREEN2, Bcolors.CEND)


def check_duplicate_code(df_data, head="库存表"):
    """
    wrap the check_duplicate function.
    :param df_data: <pandas.DataFrame>
    :param head:
    :return:
    """
    duplicate_result = check_duplicate(df_data, ["条形码", ])
    print("{0}重复条形码检测...{1}".format(head, duplicate_result))
    if "passed" not in duplicate_result:
        exit(-1)


def check_missing_cat(df_data, **kwargs):
    """
    wrap the check_missing function.
    :param df_data: <pandas.DataFrame>
    :return:
    """
    cat_result = check_missing(df_data, ["分类", ], **kwargs)
    print("总库存表未分类产品检测...{}".format(cat_result))
    if "passed" not in cat_result:
        exit(-1)


def check_error_cat(df_data, cat_range, **kwargs):
    """
    wrap the check_missing function.
    :param df_data: <pandas.DataFrame>
    :param cat_range: <list>
    :return:
    """
    cat_result = check_value_range(df_data, "分类", cat_range, **kwargs)
    print("总库存表分类错误检测...{}".format(cat_result))
    if "passed" not in cat_result:
        exit(-1)


def cal_rate_30d(data_df):
    data_df.loc[(data_df["可销库存"] + data_df["30天销量"]) != 0, "动销率"] = \
        data_df["30天销量"] / (data_df["可销库存"] + data_df["30天销量"])
    return data_df


def cal_total_cost(data_df):
    data_df.loc[data_df["可销库存"] < 0, "成本小计"] = 0
    data_df.loc[data_df["可销库存"] >= 0, "成本小计"] = data_df["成本价"] * data_df["可销库存"]
    return data_df


def dump_category(data_path, save_path, categories,
                  history_file="category_history.csv"):
    df = pd.read_excel(data_path, sheet_name=categories)
    df_list = list()
    for category in categories:
        # drop the last summary line.
        idf = df[category].loc[df[category]["条形码"] != "汇总", :]
        # pick the specified columns.
        idf = idf.loc[:, ["条形码", "产品品名", "成本价"]]
        # print warnings for the lines missing the specified columns.
        print("库存表{0}*{2}*{1}类字段缺失检测...{3}".format(Bcolors.CBOLD, Bcolors.CEND, category,
                                                   check_missing(idf, ["条形码", "产品品名"])))
        idf["分类"] = category
        df_list.append(idf)
    cat_df = pd.concat(df_list)
    check_duplicate_code(cat_df)

    cat_df.to_csv(save_path, index=False)
    if not os.path.exists(history_file):
        # create the category history file.
        cat_df.to_csv(history_file, index=False)
    else:
        # update the existed category history file.
        old_cat_df = pd.read_csv(history_file)
        merged_df = pd.concat([cat_df, old_cat_df[(~old_cat_df["条形码"].isin(cat_df["条形码"]))]])
        # merged_df = cat_df.append(old_cat_df[(~old_cat_df["条形码"].isin(cat_df["条形码"])) |
        #                                      (~old_cat_df["产品品名"].isin(cat_df["产品品名"]))])
        merged_df.to_csv(history_file, index=False)
    return cat_df


# TODO:
def guess_category():
    pass


def concat_erp(data_tables, columns, show_warnings=True):
    """
    Extract the specified columns and concatenate ERP tables, e.g. 北京1.csv, 北京2.csv, 北京3.csv,
    add up the inventory of each table, and drop the items whose inventory equal 0.
    :param data_tables: <list: str/pandas.DataFrame>: members of this list can be paths of csv
                        files or pandas.DataFrame objects.
    :param columns: <list: str>
    :param show_warnings: <boolean>
    :return: <pandas.DataFrame>
    """
    df_list = list()
    for data_table in data_tables:
        if isinstance(data_table, str):
            raw_idf = pd.read_csv(data_table, encoding="GBK")
            check_duplicate_code(raw_idf, head=data_table)
        else:
            raw_idf = data_table
        raw_idf.loc[raw_idf["30天销量"].isna(), "30天销量"] = 0
        raw_idf.loc[raw_idf["可销库存"].isna(), "可销库存"] = 0
        raw_idf = raw_idf.loc[(raw_idf["可销库存"] != 0) | (raw_idf["30天销量"] != 0)]
        df_list.append(raw_idf[columns])
    raw_df = pd.concat(df_list, )

    # define the groupby methods.
    agg_dict = dict()
    for icol in columns:
        if icol in ["可销库存", "30天销量"]:
            agg_dict.update({icol: "sum"})
        elif icol == "条形码":
            pass
        else:
            agg_dict.update({icol: "first"})

    # groupby will turn NaN in the numeric columns to 0.
    concatenated_df = raw_df.groupby("条形码", sort=False, as_index=False).agg(agg_dict)
    if show_warnings:
        check_missing(concatenated_df, ["条形码", "产品品名", "可销库存"])
    return concatenated_df


def update_inventory_table(cat_data, data_tables, update_cat=True):
    """
    update data_tables with the category/cost price data in cat_data.
    :param cat_data:
    :param data_tables:
    :param update_cat:
    :return:
    """
    if ("分类" not in data_tables.columns) and update_cat:
        data_tables["分类"] = np.NaN
    for code in data_tables["条形码"]:
        if code in cat_data["条形码"].values:
            data_tables.loc[data_tables["条形码"] == code, "成本价"] = \
                cat_data.loc[cat_data["条形码"] == code, "成本价"].values
            if update_cat:
                data_tables.loc[data_tables["条形码"] == code, "分类"] = \
                    cat_data.loc[cat_data["条形码"] == code, "分类"].values
    return data_tables
    # print(data_tables[data_tables["分类"].isna()].to_csv("no_category.csv", index=False))


def statistic_columns(data_df):
    total_invent = data_df.loc[data_df["可销库存"] >= 0, "可销库存"].sum()
    total_cost = data_df.loc[data_df["成本小计"] >= 0, "成本小计"].sum()
    result_df = data_df.append(pd.DataFrame({"条形码": ["汇总", ], "可销库存": [total_invent, ],
                                             "成本小计": [total_cost, ]}), sort=False)
    return result_df


def dataframe_format(df, columns):
    for ic in columns:
        if ic not in df.columns:
            df[ic] = np.NaN
    df = df.loc[df["条形码"] != "汇总", :]
    return df[columns]


def drop_deprecate(data_df, deprecate_data):
    """
    :param data_df: <pandas.DataFrame>
    :param deprecate_data: <list: str/pandas.DataFrame>: paths of excel file or pandas.DataFrame objects.
    :return:
    """
    if isinstance(deprecate_data, str):
        deprecate_data = pd.read_excel(deprecate_data)
    converse_df = data_df[~data_df["条形码"].isin(deprecate_data["条形码"].values)]
    return converse_df


def main():
    """
    main function
    :return:
    """
    # 分别读取北京仓,代发仓,杭州仓,广州仓ERP数据;合并各北京仓和代发仓数据,去除库存为0的产品.
    input_dir = r"d:\monica\inventory\9.16"
    invent_path = os.path.join(input_dir, "库存表.xlsx")
    wine_categories = ["畅饮", "字典", "闪购&酒局", "会员", "香槟", "小甜水", "周边",
                       "日本酒&威士忌&烈酒", "啤酒", "包材", ]
    erp_columns = ["条形码", "产品品名", "可销库存", "成本价",     # "成本小计" would be calculated later
                   "销售价", "30天销量", "国家", "产区", "葡萄品种", "规格"]
    save_columns = ["条形码", "产品品名", "动销率", "可销库存", "成本价", "成本小计",
                    "销售价", "30天销量", "国家", "产区", "葡萄品种", "规格", ]
    bj_paths = [os.path.join(input_dir, d) for d in ["北京1.csv", "北京2.csv", "北京3.csv"]]
    df_paths = [os.path.join(input_dir, d) for d in ["代发1.csv", "代发2.csv"]]
    df_bj = concat_erp(bj_paths, columns=erp_columns)
    df_df = concat_erp(df_paths, columns=erp_columns)
    df_hz = concat_erp([os.path.join(input_dir, "杭州.csv"), ], columns=erp_columns)
    df_gz = concat_erp([os.path.join(input_dir, "广州.csv"), ], columns=erp_columns)

    # 从ERP数据中去掉固定删除的条目
    df_bj = drop_deprecate(df_bj, os.path.join(input_dir, "固定删除.xlsx"))
    df_hz = drop_deprecate(df_hz, os.path.join(input_dir, "固定删除.xlsx"))
    df_gz = drop_deprecate(df_gz, os.path.join(input_dir, "固定删除.xlsx"))
    df_df = drop_deprecate(df_df, os.path.join(input_dir, "代发固定删除.xlsx"))

    # 合成各仓库ERP数据总计
    whole_erp = concat_erp([df_bj, df_hz, df_gz], columns=erp_columns)

    # 保税仓库存并入总ERP数据(需要安排在分类之前完成, 否则保税仓数据无法完成分类)
    bounded_warehouse = pd.read_excel(invent_path, sheet_name="保税仓")
    bounded_warehouse = dataframe_format(bounded_warehouse, erp_columns)
    whole_erp = concat_erp([whole_erp, bounded_warehouse], columns=erp_columns)

    # 利用库存表更新分类文件
    cat_df = dump_category(invent_path, r"category.csv", wine_categories)          # 假定分类文件在当前运行目录下

    # 检测ERP数据成本价异常
    bias_info_columns = ["条形码", "产品品名", "可销库存", "成本价_ERP", "成本价_已确认"]
    invent_df = pd.read_excel(invent_path, sheet_name="代发")
    # check_bias(df_df, invent_df, "成本价", bias_info_columns)
    check_bias(whole_erp, cat_df, "成本价", bias_info_columns).to_excel("bias.xlsx", index=False)

    # 利用分类文件更新ERP各仓库"成本价"和"分类"信息
    df_bj = update_inventory_table(cat_df, df_bj)
    df_hz = update_inventory_table(cat_df, df_hz)
    df_gz = update_inventory_table(cat_df, df_gz)
    df_df = update_inventory_table(invent_df, df_df, update_cat=False)
    whole_erp = update_inventory_table(cat_df, whole_erp)

    # 计算各仓库动销率, 成本小计
    df_bj = cal_total_cost(df_bj)
    final_bj = cal_rate_30d(df_bj).reindex(columns=save_columns).sort_values("可销库存", ascending=False)
    df_hz = cal_total_cost(df_hz)
    final_hz = cal_rate_30d(df_hz).reindex(columns=save_columns).sort_values("可销库存", ascending=False)
    df_gz = cal_total_cost(df_gz)
    final_gz = cal_rate_30d(df_gz).reindex(columns=save_columns).sort_values("可销库存", ascending=False)
    df_df = cal_total_cost(df_df)
    final_df = cal_rate_30d(df_df).reindex(columns=save_columns).sort_values("可销库存", ascending=False)
    whole_erp = cal_total_cost(whole_erp)
    whole_erp = cal_rate_30d(whole_erp).reindex(columns=save_columns + ["分类", ])

    # 输出新品列表csv文件, 手工确认新品分类
    new_cat_path = "no_category.xlsx"
    # TODO: guess the category
    whole_erp[whole_erp["分类"].isna()].to_excel(new_cat_path, index=False)
    input("等待新品分类, 完成后按回车继续...")

    # 读取完成分类的新品csv文件, 并再次更新总ERP数据分类
    new_cat = pd.read_excel(new_cat_path)
    # print(new_cat)
    final_whole = update_inventory_table(new_cat, whole_erp).sort_values("可销库存", ascending=False)
    check_missing_cat(final_whole, level="fatal")
    check_error_cat(final_whole, wine_categories, level="fatal")

    # 挑出打价姐,牛肉分类.
    store_codes = pd.read_excel(invent_path, sheet_name="打价姐店").loc[:, "条形码"]
    beef_codes = pd.read_excel(invent_path, sheet_name="牛肉").loc[:, "条形码"]
    store_df = final_whole.loc[final_whole["条形码"].isin(store_codes), :].reindex(columns=save_columns)
    beef_df = final_df.loc[final_df["条形码"].isin(beef_codes), :].reindex(columns=save_columns)

    # 输出各分类库存表
    with pd.ExcelWriter('库存表_{}.xlsx'.format(datetime.now().strftime("%F"))) as invent_writer:
        for cat in wine_categories:
            statistic_columns(final_whole.loc[final_whole["分类"] == cat, :].reindex(
                columns=save_columns)).to_excel(invent_writer, sheet_name=cat, index=False)
        statistic_columns(store_df).to_excel(invent_writer, sheet_name="打价姐店", index=False)
        statistic_columns(final_bj).to_excel(invent_writer, sheet_name="北京仓", index=False)
        statistic_columns(final_hz).to_excel(invent_writer, sheet_name="杭州仓", index=False)
        statistic_columns(final_gz).to_excel(invent_writer, sheet_name="广州仓", index=False)
        statistic_columns(beef_df).to_excel(invent_writer, sheet_name="牛肉", index=False)
        statistic_columns(final_df).to_excel(invent_writer, sheet_name="代发", index=False)
        bounded_warehouse.to_excel(invent_writer, sheet_name="保税仓", index=False)

    # 计算各仓总计


if __name__ == "__main__":
    pd.set_option("display.max_columns", 20)

    # concat_invent(r"d:\monica\inventory\tables\库存表.xlsx", "北京仓")
    # dump_category(r"d:\monica\inventory\tables\库存表.xlsx", r"d:\monica\inventory\category.csv")
    main()
