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
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler

LOGGER = get_logger(__name__)

@dataclass

class Params:
    '''
    All parameters used throughout the code
    '''
    sigma = 0.2 # determines how significant the weight a variable is
    scale_range = (0, 20)
    mismatch_weight = 10
    virtual_weight = 10
    test_size = 0.2
    num_layers = 2
    hidden_dim = 64
    dropout = 0.05
    batch_size = 32
    lr = 0.001
    random_state = 25

params = Params()

def import_base_data():
    branch_df = pd.read_excel('data/Branch_Level_Dataset.xlsx')
    member_df = pd.read_csv('data/Member_Level_Dataset.csv')
    return branch_df, member_df

def process_data(branch_df, member_df):
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

    # set EOM_TRANS_DATE to datetime
    member_df["EOM_TRANS_DATE"] = member_df["EOM_TRANS_DATE"].astype("datetime64")

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
    # average was selected for these columns because when insights are being made, the total amount of accounts a member has is not as important as the average amount of accounts a member has for EOM analysis and Branch performance
    agg_dict.update({col: 'mean' for col in avg_columns})

    grouped_members_eom = member_df.groupby(['BranchCategory', 'EOM_TRANS_DATE']).agg(agg_dict).reset_index()
    group_members = member_df.groupby(['BranchCategory']).agg(agg_dict).reset_index()

    return group_members, grouped_members_eom, branch_df

def group_branch_data(df):
    grouped_df = df.groupby('BranchCategory').agg({
    'County': 'last',
    'ATM': 'sum',
    'Bill Payment': 'sum',
    'Cash': 'sum',
    'Draft': 'sum',
    'ACH': 'sum',
    'Fee': 'sum',
    'Credit/Debit Card': 'sum',
    'Home Banking': 'sum',
    'Dividend': 'sum',
    'EOM_TRANS_DATE': 'count',
    }).rename(columns={'EOM_TRANS_DATE': 'Total_Transactions'}).reset_index()
    return grouped_df

def merge_dataframes(df1, df2):
    merged_df = pd.merge(df1, df2, on=['EOM_TRANS_DATE', 'BranchCategory'])
    return merged_df

#Weighting and Scaling Merged Data
def weigh_scale_data(df):
    scaler = StandardScaler()
    columns_to_scale = ['ATM', 'Bill Payment', 'Cash', 'Draft', 'ACH', 'Fee', 'Credit/Debit Card', 'Home Banking', 'Dividend', 'ATMCount', 'BillPaymentCount', 'CashCount', 'DraftCount', 'ACHCount', 'FeeCount', 'Credit_DebitCount', 'Home_Banking', 'WireCount', 'DividendCount', 'n_accts', 'n_checking_accts', 'n_savings_accts', 'n_open_loans', 'n_open_cds', 'n_open_club_accts', 'n_open_credit_cards']
    scaled_df = df.copy()
    scaled_df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

    branch_weights = {
        'ATM': 1, # Standard transactions
        'Bill Payment': 1 + 1.5*params.sigma, # Slightly more important due to its link to loans
        'Cash': 1, # Standard transactions
        'Draft': 1, # Standard transactions
        'ACH': 1 + 2*params.sigma, # Primary banking indicator, more important
        'Fee': -1 - 1*params.sigma, # Negative impact
        'Credit/Debit Card': 1 + 1*params.sigma, # Standard transactions
        'Home Banking': 1 + 1*params.sigma, # Indicates engagement but not directly profitability
        'Dividend': 1 + 3*params.sigma, # High importance for profitability
        'n_checking_accts': 1 + 1*params.sigma, # Standard account type
        'n_savings_accts': 1 + 2*params.sigma, # Indicator of stored funds
        'n_open_loans': 1 + 1.5*params.sigma, # Important for indebted customer base
        'n_open_cds': 1 + 3*params.sigma, # High-value accounts, significant for profitability
        'n_open_club_accts': 1, # Standard account type
        'n_open_credit_cards': 1 + 1*params.sigma, # Indicative of spending but not direct profitability
        'ATMCount': 1, # Standard transaction type
        'BillPaymentCount': 1 + 2*params.sigma, # Linked to loans, hence more important
        'CashCount': 1, # Standard transaction type
        'DraftCount': 1, # Standard transaction type
        'ACHCount': 1 + 3*params.sigma, # Primary banking indicator, more important
        'FeeCount': -1 - 1*params.sigma, # Negative impact
        'Credit_DebitCount': 1 + 1*params.sigma, # Standard transaction type
        'Home_Banking': 1 + 1*params.sigma, # Indicates engagement but not directly profitability
        'WireCount': 1 + 1.5*params.sigma, # Slightly more important due to larger transactions
        'DividendCount': 1 + 3*params.sigma, # High importance for profitability
        }

    weighted_df = scaled_df.copy()
    columns_for_scoring = columns_to_scale
    
    for column, weight in branch_weights.items():
        weighted_df[column] *= weight

    weighted_df['Branch_Score'] = weighted_df[columns_for_scoring].sum(axis=1)

    weighted_df[['BranchCategory', 'Branch_Score']].sort_values(by='Branch_Score', ascending=False).reset_index()

    return weighted_df

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
    branch_df, member_df = import_base_data()
    
    # Data Prociessing & Branch Grouping
    group_members, group_members_eom, branch_df = process_data(branch_df, member_df)
    grouped_branches = group_branch_data(branch_df)

    st.markdown("""
                ### Data Import
    """)
    
    with st.expander("See Data Import"):
        tab_member, tab_branch, tab_grouped = st.tabs(['Member Data','Branch Data', 'Grouped Branch Data'])

        with tab_member:
            st.markdown(f"""
                        #### Member Data

                        {member_df.shape}
                    """)
            st.dataframe(member_df.head())
            st.dataframe(member_df.describe())

        with tab_branch:
            st.markdown(f"""
                        #### Branch Data

                        {branch_df.shape}
                    """)
            st.dataframe(branch_df)
            st.dataframe(branch_df.describe())

        with tab_grouped:
            st.markdown(f"""
                        #### Grouped Branch Data

                        {grouped_branches.shape}
                        """)
            st.dataframe(grouped_branches)
            st.dataframe(grouped_branches.describe())

    # Member DataFrame All Possible Branches
    # st.text("Member DataFrame All Possible Branches")
    # st.dataframe(member_df['BranchCategory'].unique())
    
    st.markdown(f"""
                #### Post Import Size

                Member Data = {member_df.shape}

                Branch Data = {branch_df.shape}

                Grouped Branch Data = {grouped_branches.shape}
    """)
    
    # Data Pre-Processing Section
    st.markdown("### Pre-Processing")

    with st.expander("See Data Processing"):
        # Before Processing
        st.markdown(f"""
                    #### Member Data Processed
                    
                    Member Size = {member_df.shape}
        """)
        st.dataframe(member_df.head())

        # After Processing w/ Dates
        st.markdown(f"""
                    #### Grouped Member Data (Branch & Date)
                    
                    Grouped Size = {group_members_eom.shape}
        """)
        st.dataframe(group_members_eom)

        # After Processing w/o Dates
        st.markdown(f"""
                    #### Grouped Member Data
                    
                    Grouped Size = {group_members.shape}
        """)
        st.dataframe(group_members)

    st.markdown(f"""
                #### Post Processing Size

                ##### Member Data

                Member Data = {member_df.shape}

                Grouped Member Data (Branch & Date) = {group_members_eom.shape}

                Grouped Member Data = {group_members.shape}

                ##### Branch Data

                Branch Data = {branch_df.shape}

                Grouped Branch Data = {grouped_branches.shape}

    """)
        
    st.markdown("### Exploratory Analysis")

    with st.expander("See Exploratory Analysis"):
        tab_members, tab_branch = st.tabs(['Member Data', 'Branch Data'])
        
        with tab_members:
            st.markdown("#### Member Data")
            st.dataframe(member_df.describe())
            fig = create_corr_matrix_heatmap(member_df)
            st.plotly_chart(fig)

            st.markdown("#### Grouped Member Data (Branch & Date)")
            st.dataframe(group_members_eom.describe())
            fig = create_corr_matrix_heatmap(group_members_eom)
            st.plotly_chart(fig)

            st.markdown("#### Grouped Member Data")
            st.dataframe(group_members.describe())
            fig = create_corr_matrix_heatmap(group_members)
            st.plotly_chart(fig)

        with tab_branch:
            st.markdown("#### Branch Data")
            st.dataframe(branch_df.describe())
            fig = create_corr_matrix_heatmap(branch_df)
            st.plotly_chart(fig)

            st.markdown("#### Grouped Branch Data")
            st.dataframe(grouped_branches.describe())
            fig = create_corr_matrix_heatmap(grouped_branches)
            st.plotly_chart(fig)

    st.markdown("### Branch & Member Data Time Series Alignment")

    with st.expander("See Time Series Alignment"):
        col1, col2 = st.columns(2)
        with col1:
            st.write("Branch Data Columns")
            st.dataframe(branch_df.columns)
            # print the datatypes of the columns
            st.write("Branch Data Types")
            st.dataframe(branch_df.dtypes)
        
        with col2:
            st.write("Member Data Columns")
            st.dataframe(group_members_eom.columns)
            # print the datatypes of the columns
            st.write("Member Data Types")
            st.dataframe(group_members_eom.dtypes)

    # merging branch and member by EOM_TRANS_DATE BranchCategory
        merged_df = merge_dataframes(group_members_eom, branch_df)
        st.dataframe(merged_df)
    
    st.markdown(f"""
                #### Post Merge Analysis
                
                ##### Branch Data
                
                Branch Data = {branch_df.shape}
                
                ##### Grouped Member Data (Branch & Date)
                
                Grouped Member Data (Branch & Date) = {group_members_eom.shape}
                
                ##### Merged Data
                
                Merged Data = {merged_df.shape}
    """)

    st.markdown("### Weighing Scaling & Merging Data")
    weighed_df = weigh_scale_data(merged_df)
    st.dataframe(weighed_df)
    
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
