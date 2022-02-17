import datetime
import json

import pandas as pd
import numpy as np


VERSION = "0.00"


class BGDataset:
    version = "0.00"
    created = None
    updated = None
    paitients = []

    _set_id = set()

    def __init__(self):
        self.created = self.__now()
        self.updated = self.created
        self.paitients = []
        self._set_id = set()

    @staticmethod
    def __now():
        time_format = "%Y-%m-%d %H:%M"
        t_delta = datetime.timedelta(hours=9)
        JST = datetime.timezone(t_delta, 'JST')
        dt = datetime.datetime.now(JST)
        return dt.strftime(time_format)

    def add_paitient(self, bg_paitient):
        if bg_paitient.id in self._set_id:
            raise Exception("ID duplicated")
        else:
            self.paitients.append(bg_paitient)
            self._set_id.add(bg_paitient.id)
            self.updated = self.__now()

    def status(self):
        print("created:", self.created)
        print("updated:", self.updated)
        print("paitient num:", len(self.paitients))
        id_list = [x.id for x in self.paitients]
        print("min id:", min(id_list))
        print("max id:", max(id_list))

    def to_dict(self):
        return {
            "version": self.version,
            "created": self.created,
            "updated": self.updated,
            "paitient_data_list": [x.to_dict() for x in self.paitients]
        }

    def to_json(self, path):
        text = json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
        with open(path, "w") as f:
            # json.dump(self.to_dict(), f, indent=2)
            f.write(text)

    @staticmethod
    def from_dict(data):
        instance = __class__()
        instance.version = data["version"]
        instance.created = data["created"]
        instance.updated = data["updated"]
        instance.paitients = [BGPaitient.from_dict(x) for x in data["paitient_data_list"]]
        return instance

    @staticmethod
    def from_json(path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return __class__.from_dict(data)


class BGPaitient:
    id = None
    source = None
    source_id = None
    sex = None
    age = None
    hba1c = None

    def __init__(self, id=id, source=source, source_id=source_id):
        self.id = id
        self.source = source
        self.source_id = source_id

        self.insulin_injection = []
        self.insulin_pump = []

        self.bg_data = []
        self.meal_data = []
        self.insulin_data = []
        self.exercise_data = []

    def add_bg(self, df):
        """get blood glucose dataset from Pandas DataFrame

        :pram df: Pandas DataFrame. Column 1 is datetime, column 2 is blood glucose value
        :return: python dictionary data"""

        data = []
        for v in df.itertuples():
            data.append({"datetime": v[1].strftime("%Y-%m-%d %H:%M"), "blood_glucose_value": v[2]})
        self.bg_data = data

    def add_meal(self, df):
        """get blood glucose dataset from Pandas DataFrame

        :pram df: Pandas DataFrame. Column 1 is date, 2 meal type, 3 time, 4 carbohydrate, 5 dietary fiber, 6 suger mass
        :return: python dictionary data"""

        df = df.dropna().copy()
        df["datetime"] = pd.to_datetime(df.iloc[:, 0].astype(str) + " " + df.iloc[:, 2].astype(str))
        data = []
        for v in df.itertuples():
            data.append({"datetime": v.datetime.strftime("%Y-%m-%d %H:%M"), "carbohydrate": v[4], "dietary_fiber": v[5],
                         "suger_mass": v[6]})
        self.meal_data = data

    def add_base_info(self, df, insulin_method="injection"):
        if insulin_method == "injection":
            self._add_base_info_injection(df)
        elif insulin_method == "pump":
            self._add_base_info_pump(df)

    def _add_base_info_injection(self, df):
        """create paitient data from Pandas DataFrame. Only use insulin injection paitient.

        :param df: Pandas DataFrame. Column 1 is jikougata insulin, 2 is chosokou insulin, 3 is sex, 4 is HbA1c, 5 is age.
        :return: python dictionary data"""

        self.sex = str(df.iloc[0, 3 - 1])
        self.age = int(df.iloc[0, 5 - 1])
        self.hba1c = float(df.iloc[0, 4 - 1])
        long_acting_insulin_name, long_acting_insulin_unit = df.iloc[:2, 1 - 1]
        super_fast_acting_insulin_name, super_fast_acting_insulin_unit = df.iloc[:2, 2 - 1]
        super_fast_acting_insulin_unit = list(map(int, super_fast_acting_insulin_unit.split("*")))

        self.insulin_injection = [
            {
                "insulin_name": long_acting_insulin_name,
                "insulin_type": "持効型インスリン",
                "injection_timing": "1day",
                "day_unit": long_acting_insulin_unit,
            },
            {
                "insulin_name": super_fast_acting_insulin_name,
                "insulin_type": "超即効型インスリン",
                "injection_timing": "before_meals",
                "breakfast_unit": super_fast_acting_insulin_unit[0],
                "lunch_unit": super_fast_acting_insulin_unit[1],
                "dinner_unit": super_fast_acting_insulin_unit[2],
            }
        ]

    def _add_base_info_pump(self, df):
        """create paitient data from Pandas DataFrame. Only use insulin pomp paitient.

        :param df: Pandas DataFrame. Column 1-2 is insulin pomp data, 3 is sex, 4 is HbA1c, 5 is age.
        :return: python dictionary data"""

        self.sex = str(df.iloc[0, 3 - 1])
        self.age = int(df.iloc[0, 5 - 1])
        self.hba1c = float(df.iloc[0, 4 - 1])
        pump_list = []
        for v in df.iloc[:, [1 - 1, 2 - 1]].dropna().itertuples():
            pump_list.append({"time": v[1], "unit": v[2]})

        self.insulin_pump = pump_list

    def status(self):
        print("paitient id:", self.id)
        print("source:", self.source)
        print("source id:", self.source_id)
        print("age:", self.age)
        print("sex:", self.sex)
        print("HbA1c:", self.hba1c)
        print("BG data count:", len(self.bg_data))
        print("Meal data count:", len(self.meal_data))
        print("Insulin injection:", self.insulin_injection != [])
        print("Insulin pump:", self.insulin_pump != [])

    def to_dict(self):
        return {
            "paitient_id": self.id,
            "source_id": self.source_id,
            "source": self.source,
            "age": self.age,
            "sex": self.sex,
            "hba1c": self.hba1c,
            "insulin_pump": self.insulin_pump,
            "insulin_injection": self.insulin_injection,
            "blood_glucose_data": self.bg_data,
            "meal_data": self.meal_data,
            "insulin_data": self.insulin_data,
            "exercise_data": self.exercise_data,
        }

    @staticmethod
    def from_dict(data):
        instance = __class__()
        instance.id = data["paitient_id"]
        instance.source_id = data["source_id"]
        instance.source = data["source"]
        instance.age = data["age"]
        instance.sex = data["sex"]
        instance.hba1c = data["hba1c"]
        instance.insulin_injection = data["insulin_injection"]
        instance.insulin_pump = data["insulin_pump"]
        instance.bg_data = data["blood_glucose_data"]
        instance.meal_data = data["meal_data"]
        instance.insulin_data = data["insulin_data"]
        instance.exercise_data = data["exercise_data"]
        return instance

    def get_bg(self):
        df1 = pd.DataFrame(self.bg_data)
        df1["datetime"] = pd.to_datetime(df1["datetime"])
        df1["blood_glucose_value"] = df1["blood_glucose_value"].astype(float)
        return df1

    def get_meal(self):
        df1 = pd.DataFrame(self.meal_data)
        df1["datetime"] = pd.to_datetime(df1["datetime"])
        return df1

    def get_base_info(self):
        if self.sex == "male":
            sex_dgt = 1
        elif self.sex == "female":
            sex_dgt = 0
        else:
            raise Exception("sex data invalid")
        return pd.DataFrame({
            "age": [self.age],
            "sex": [sex_dgt],
            "HbA1c": [self.hba1c],
        })


def compliment_bg(df, min, silent=False):
    gap_max_rate = 1.8  # 15分ごとのデータの場合、15*1.8=27分の空きをギャップと認識
    time_idx = 0
    val_idx = 1

    arr1 = df.loc[:, ["datetime", "blood_glucose_value"]].to_numpy()
    arr2 = []
    for i in range(len(arr1)-1):
        arr2.append(arr1[i])
        current_dt = arr1[i][0]
        next_dt = arr1[i+1][0]
        gap = (next_dt - current_dt).total_seconds() / 60
        if gap > min * gap_max_rate:
            if not silent:
                print("gap is found [{}] to [{}]. value is complimented. ({} to {}).".format(arr1[i][0], arr1[i+1][0], arr1[i][1], arr1[i+1][1]))
            comp_num = int(gap / min)
            sum_diff = arr1[i+1][1] - arr1[i][1]
            unit_diff = sum_diff / comp_num
            for j in range(comp_num-1):
                arr2.append(np.array([current_dt + datetime.timedelta(minutes=min*(j+1)), arr1[i][1]+unit_diff*(j+1)]))
    arr2.append(arr1[-1])

    return pd.DataFrame(arr2, columns=("datetime", "blood_glucose_value"))


def add_time_course(bg_df, meal_df):

    bg_df1 = bg_df.copy(deep=True)
    meal_df1 = meal_df.copy(deep=True)
    start_dt = bg_df["datetime"][0]
    time_course = []
    for i, dt in enumerate(bg_df1["datetime"]):
        time_course.append((dt - start_dt).total_seconds() / (60 * 60 * 24))
    bg_df1.insert(1, "time_course", time_course)
    time_course = []
    for i, dt in enumerate(meal_df1["datetime"]):
        time_course.append((dt - start_dt).total_seconds() / (60 * 60 * 24))
    meal_df1.insert(1, "time_course", time_course)

    return bg_df1, meal_df1


def split_days(bg_df, meal_df, days=2):

    bg_df_list = []
    ml_df_list = []
    for i in range(0, 10, days):
        bg_df1 = bg_df[bg_df["time_course"] >= i]
        bg_df1 = bg_df1[bg_df1["time_course"] < i+days]
        bg_df_list.append(bg_df1)
        ml_df1 = meal_df[meal_df["time_course"] >= i]
        ml_df1 = ml_df1[ml_df1["time_course"] < i+days]
        ml_df_list.append(ml_df1)

    return bg_df_list, ml_df_list


if __name__ == "__main__":
    json_path = "test/BG_dataset_20220204.json"
    data = BGDataset.from_json(json_path)
    for paitient in data.paitients:
        bg_df = paitient.get_bg()  # 血糖値データを抽出
        meal_df = paitient.get_meal()  # 食事データを抽出
        bg_df = compliment_bg(bg_df, 15, True)  # データの空白を埋める
        bg_df, meal_df = add_time_course(bg_df, meal_df)  # データの中に経過時間(日)を付与
        bg_df_list, meal_df_list = split_days(bg_df, meal_df, 2)  # 2日ごとのデータに分割

        print(bg_df_list[0])
        break
