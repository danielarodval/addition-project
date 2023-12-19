# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import streamlit as st
from streamlit.logger import get_logger
import pandas as pd

LOGGER = get_logger(__name__)


def run():
    st.set_page_config(
        page_title="Insights",
        page_icon="💹",
    )

    st.write("# Addition Financial Project")

    st.sidebar.success("Select a demo above.")

    st.markdown(
        """
        ## Data Import
    """
    )
    branch_df = pd.read_excel('data/Branch_Level_Dataset.xlsx')
    member_df = pd.read_csv('data/Member_Level_Dataset.csv')
    st.dataframe(branch_df.head())

if __name__ == "__main__":
    run()
