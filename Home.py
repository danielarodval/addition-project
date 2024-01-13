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
from datetime import datetime
import numpy as np
from uszipcode import SearchEngine
import plotly.express as px

LOGGER = get_logger(__name__)

def create_corr_matrix_heatmap(df):
    corr_matrix = df.corr()
    fig = px.imshow(corr_matrix,
                    labels=dict(x="Feature", y="Feature", color="Correlation"),
                    x=corr_matrix.columns,
                    y=corr_matrix.columns)
    
    fig.update_layout(title="Correlation Matrix",
                      xaxis_nticks=len(corr_matrix.columns),
                      yaxis_nticks=len(corr_matrix.columns),
                      xaxis_title="Feature",
                      yaxis_title="Feature")
    return fig

def run():
    st.set_page_config(
        page_title="Insights",
        page_icon="💹",
    )

    st.write("# Addition Financial Project")

    st.sidebar.success("Select a demo above.")
    # Data Import Section
    branch_df = pd.read_excel('data/Branch_Level_Dataset.xlsx')
    member_df = pd.read_csv('data/Member_Level_Dataset.csv')
    
    grouped_branches = member_df.groupby(by=["BranchCategory"]).sum(numeric_only=True)

    st.markdown(
        """
        ## Data
        ### Import
    """
    )
    with st.expander("See Data Import"):
        tab_branch, tab_member = st.tabs(['Branch Data','Member Data'])
        with tab_branch:
            st.dataframe(branch_df.head())
        with tab_member:
            st.dataframe(member_df.head())
        st.dataframe(grouped_branches.head())

    st.markdown(
        """
        ### Pre-Processing
    """
    )

    with st.expander("See Data Processing"):
        county_map = {
        'Addition Financial Arena': 'Orange',
        'Apopka': 'Orange',
        'Boone High School': 'Orange',
        'Colonial High School': 'Orange',
        'Downtown Campus': 'Orange',
        'East Orlando': 'Orange',
        'Edgewater High School': 'Orange',
        'Lake Nona': 'Orange',
        'MetroWest': 'Orange',
        'Mills': 'Orange',
        'Oak Ridge High School': 'Orange',
        'Ocoee High School': 'Orange',
        'Pine Hills': 'Orange',
        'South Orlando': 'Orange',
        'Timber Creek High School': 'Orange',
        'UCF Campus': 'Orange',
        'UCF Commons': 'Orange',
        'Winter Garden': 'Orange',
        'Altamonte Springs': 'Seminole',
        'Fern Park': 'Seminole',
        'Lake Brantley High School': 'Seminole',
        'Lake Howell High School': 'Seminole',
        'Lake Mary': 'Seminole',
        'Longwood': 'Seminole',
        'Oviedo': 'Seminole',
        'Sanford': 'Seminole',
        'Seminole State': 'Seminole',
        'Clermont': 'Lake',
        'Eustis': 'Lake',
        'Leesburg': 'Lake',
        'Kissimmee': 'Osceola',
        'Poinciana High School': 'Osceola',
        'St. Cloud': 'Osceola',
        'St. Cloud High School': 'Osceola',
        'The Loop': 'Osceola',
        'Merritt Island': 'Brevard',
        'Orange City': 'Volusia',
        'Poinciana': 'Polk',
        'Virtual Branch': None  # Virtual Branch does not have a county
        }

        branch_df['County'] = branch_df['BranchCategory'].map(county_map)
        member_df['Branch_County'] = member_df['BranchCategory'].map(county_map)

        unique_zips = member_df['address_zip'].unique()

        search = SearchEngine()

        zip_to_county_map = {
            zip_code: (search.by_zipcode(zip_code).county if search.by_zipcode(zip_code) else np.nan)
            for zip_code in unique_zips
        }

        member_df['address_county'] = member_df['address_zip'].map(zip_to_county_map)
        member_df['address_county'] = member_df['address_county'].str.replace(' County', '', regex=False)

        missing_county_members = member_df[pd.isna(member_df['address_county'])]
        missing_zips = missing_county_members['address_zip'].unique()

        unique_county = member_df['address_county'].unique()
        transaction_columns = ['ATMCount', 'BillPaymentCount', 'CashCount', 'DraftCount', 'ACHCount', 'FeeCount', 'Credit_DebitCount', 'Home_Banking', 'WireCount', 'DividendCount']
        avg_columns = ['n_accts', 'n_checking_accts', 'n_savings_accts', 'n_open_loans', 'n_open_cds', 'n_open_club_accts', 'n_open_credit_cards']

        # Define the aggregation dictionary
        agg_dict = {col: 'sum' for col in transaction_columns}
        agg_dict.update({col: 'mean' for col in avg_columns})

        grouped_br_eom_df = member_df.groupby(['BranchCategory', 'EOM_TRANS_DATE']).agg(agg_dict).reset_index()
        grouped_br_df = member_df.groupby(['BranchCategory']).agg(agg_dict).reset_index()

        st.markdown(f"""Grouped Data Processed
                    Grouped Size = {grouped_br_df.shape}
                    """)
        st.dataframe(grouped_br_df)
        st.markdown(f"""Grouped Data (Branch & Month) Processed
                    Grouped Size = {grouped_br_eom_df.shape}
                    """)
        st.dataframe(grouped_br_eom_df)

        st.markdown(f"""Member Data Processed
                    Member Size = {member_df.shape}
                    """)
        st.dataframe(member_df.head())
        
    st.markdown("### Exploratory Analysis")

    st.markdown("#### Grouped Data")
    #st.dataframe(grouped_br_df.describe())
    #fig = create_corr_matrix_heatmap(grouped_br_df)
    #st.plotly_chart(fig)

    st.markdown("#### Grouped (Branch & Month) Data")
    st.dataframe(grouped_br_eom_df.describe())
    fig = create_corr_matrix_heatmap(grouped_br_eom_df)
    st.plotly_chart(fig)

    st.markdown("#### Member Data")
    st.dataframe(member_df.describe())
    fig = create_corr_matrix_heatmap(member_df)
    st.plotly_chart(fig)



    with st.expander("Notes"):
        st.text("""
                correlation (matrix or pairplot)    
                stats(mean,median, etc)
                            
                residuals vs fitted [depends on model]
                            
                feature importance (ranking, score, coef (confusion matrix), etc) [depends on model]

                Branch Data (Date Column)
                Member Data (More Date Column)

                Member Data - Groupby Date then Groupby Branch Category (sum all attributes) (keep unique member transactions per day)
                Merge Datasets based on date

                Grouping Member Data, youll find multiple zip codes, getting a unique count for each zip in each branch is good, that allows us to see how many external zips come to this branch

                Find all zips that dont equal branch zip, count each, greatest number means that zip could yield a new location""")

print(datetime.now())
if __name__ == "__main__":
    run()
