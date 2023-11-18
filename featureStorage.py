from typing import Any
from feast import FeatureStore
from langchain.prompts import PromptTemplate, StringPromptTemplate

# Because gunicorn dont fully support windows, this script do not work on windows

feast_repo_path = "./my_feature_repo/feature_repo/"
store = FeatureStore(repo_path=feast_repo_path)

template = """Given the driver's up to date stats, write them note relaying those stats to them.
If they have a conversation rate above .5, give them a compliment. Otherwise, make a silly joke about chickens at the end to make them feel better

Here are the drivers stats:
Conversation rate: {conv_rate}
Acceptance rate: {acc_rate}
Average Daily Trips: {avg_daily_trips}

Your response:"""
prompt = PromptTemplate.from_template(template)


class FeastPromptTemplete(StringPromptTemplate):
    def format(self, **kwargs: Any) -> str:
        driver_id = kwargs.pop("driver_id")
        feature_vector = store.get_online_features(
            features=[
                "driver_hourly_stats:conv_rate",
                "driver_hourly_stats:acc_rate",
                "driver_hourly_stats:avg_daily_trips",
            ],
            entity_rows=[{"driver_id":driver_id}],
        ).to_dict()
        kwargs["conv_rate"] = feature_vector["conv_rate"][0]
        kwargs["acc_rate"] = feature_vector["acc_rate"][0]
        kwargs["avg_daily_trips"] = feature_vector["avg_daily_trips"][0]
        return super().format(**kwargs)
    

prompt_template = FeastPromptTemplate(input_variables=["driver_id"])

print(prompt_template.format(driver_id=1001))