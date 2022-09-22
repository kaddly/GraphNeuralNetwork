import pandas as pd
import os


def convert_age(age_str):
    if age_str == u'-1':
        return 0
    elif age_str == u'15岁以下':
        return 1
    elif age_str == u'16-25岁':
        return 2
    elif age_str == u'26-35岁':
        return 3
    elif age_str == u'36-45岁':
        return 4
    elif age_str == u'46-55岁':
        return 5
    elif age_str == u'56岁以上':
        return 6
    else:
        return -1


def convert_id(id_str):
    return 'u_' + str(id_str)


def convert_sku_id(idstr):
    return 'i_' + str(idstr)


def convert_action_user(idstr):
    idstr = int(idstr)
    return 'u_' + str(idstr)


def convert_action_item(idstr):
    return 'i_' + str(idstr)


def data_procession(data_dir=os.path.join('./', 'data')):
    USER = pd.read_csv(os.path.join(data_dir, 'JData_User.csv'), encoding='gbk')
    ITEM = pd.read_csv(os.path.join(data_dir, 'JData_Product.csv'), encoding='gbk')
    ACTION = pd.read_csv(os.path.join(data_dir, 'JData_Action_201602.csv'), encoding='gbk')

    user = USER.copy()
    user['age'] = user['age'].map(convert_age)
    user['user_id'] = user['user_id'].map(convert_id)
    age_df = pd.get_dummies(user["age"], prefix="age")
    sex_df = pd.get_dummies(user["sex"], prefix="sex")
    user_lv_df = pd.get_dummies(user["user_lv_cd"], prefix="user_lv_cd")
    data_user = pd.concat([user['user_id'], age_df, sex_df, user_lv_df], axis=1)

    item = ITEM.copy()
    item['sku_id'] = item['sku_id'].map(convert_sku_id)
    a1_df = pd.get_dummies(item["a1"], prefix="a1")
    a2_df = pd.get_dummies(item["a2"], prefix="a2")
    a3_df = pd.get_dummies(item["a3"], prefix="a3")
    cate_df = pd.get_dummies(item['cate'], prefix='cate')
    brand_df = pd.get_dummies(item['brand'], prefix='brand')
    data_item = pd.concat([item['sku_id'], a1_df, a2_df, a3_df, cate_df, brand_df], axis=1)

    action = ACTION.copy()
    action = action[action['type'] == 6]
    action.drop(['time', 'model_id', 'type', 'cate', 'brand'], axis=1, inplace=True)
    # 筛选用户,去掉点击数比较多的爬虫用户,去掉点击数比较少的冷启用户
    # x=action.groupby('user_id').count()['sku_id']
    # users=list(x[(x.values>150)&(x.values<200)].index)
    action.reset_index()
    action = action.drop_duplicates()
    action['user_id'] = action['user_id'].map(convert_action_user)
    action['sku_id'] = action['sku_id'].map(convert_action_item)

    data_user_t = data_user[data_user['user_id'].isin(list(action['user_id']))]
    data_user_t.rename(columns={'user_id': 'node_id'}, inplace=True)
    data_user_t.to_csv('./data/user_features.csv', index=False)

    data_item_t = data_item[data_item['sku_id'].isin(list(action['sku_id']))]
    data_item_t.rename(columns={'sku_id': 'node_id'}, inplace=True)
    data_item_t.to_csv('./data/item_features.csv', index=False)

    node_features = pd.concat([data_user_t, data_item_t], keys='node_id', ignore_index=True)
    node_features.fillna(0, inplace=True)

    node_features.to_csv('./data/node_features.csv', index=False)

    action.to_csv('./data/data_action.csv', index=False)


if __name__ == '__main__':
    data_procession()
