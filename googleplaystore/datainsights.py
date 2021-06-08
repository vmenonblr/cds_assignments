import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import numpy as np



class DataInsigthFinder():

    def __init__(self, df):
        self.data_frame = df
        plt.rc('figure', figsize=(10, 8))

    def __extract_year(self,x):
        return int(x[-4:])

    def plot_scatter_size_price(self):
        flt1 = self.data_frame['Size'] > 0
        flt2 = self.data_frame['Installs'] > 0
        flt3 = self.data_frame['Price'] > 0
        # Only plotting for apps which are not free and see correlation between size and price
        df_filtered = self.data_frame.where(flt1 & flt2 & flt3)
        sns_plot = sns.scatterplot(x='Size', y='Price', data=df_filtered)
        #sns_plot.set_xticks(1,100)
        fig = sns_plot.get_figure()
        fig.savefig('plots/scattersizeprice.png')
        plt.clf()

    def plot_scatter_rating_installs_cat(self):
        try:
            BAR_WIDTH = 0.4
            df_filtered = self.data_frame[self.data_frame['Category'] != '1.9']
            # Aggregate average rating and installs
            df_plot = df_filtered.groupby('Category').agg({'Rating':'mean','Installs':'mean'}).reset_index()
            df_plot.index = df_plot['Category']
            plt.bar((range(len(df_plot.index))), df_plot['Rating'].values, width=BAR_WIDTH, color='b', label='Avg. Rating')
            # Convert avg installs to millions so that scales will match
            l_installs = df_plot['Installs'].values/(10**6)
            l_x = []
            for x in (range(len(df_plot.index))):
                l_x.append(x+BAR_WIDTH)
            plt.bar(l_x, l_installs, width=BAR_WIDTH, color='r', label='Avg. Installs-millions')
            plt.xticks(range(len(df_plot.index)), df_plot.index, rotation='vertical')
            plt.legend()
            plt.savefig('plots/catinstallratings.png')
            plt.clf()
            print('Done')

        except:
            print("Unexpected Error {0}".format(sys.exc_info()[0]))

    def plot_count_cat_year_apps(self):
        try:

            self.data_frame['Year'] = [x[-4:] for x in self.data_frame['Last Updated'].tolist()]
            # Find the average rating group by category and year
            s_cat = self.data_frame.groupby(['Category','Year']) .agg({'Rating': 'mean'}).reset_index()
            df_plot = pd.DataFrame(s_cat)
            # Filter only apps with high rating
            df_plot = df_plot[(df_plot['Category'] != '1.9') & (df_plot['Rating'] > 4.0)]
            df_plot.index = df_plot['Category']
            # Plot count in years for categories which had consistently had high rating
            # NOT CLEAR ON BINNING OF COUNT PLOTS. WILL NEED TO DISCUSS!!!!!
            sns_plot = sns.countplot(x='Category',  data=df_plot)
            plt.xticks(range(len(df_plot.index)), df_plot.index, rotation='vertical')
            fig = sns_plot.get_figure()
            fig.savefig('plots/countratingcatyear.png')
            plt.clf()

        except:
            print("Unexpected Error {0}".format(sys.exc_info()[0]))

    def find_highest_paid_app(self):
        # Get Apps with good rating
        df_filtered = self.data_frame[(self.data_frame['Category'] != '1.9') & (self.data_frame['Rating'] > 4.5)]
        # Sort based in Price
        df_sorted = df_filtered.sort_values(by=['Price'], ascending=False)
        #Get highest prices app
        app_name = df_sorted.iloc[0]['App']
        return app_name

    def find_top_rated_app_geniuine(self):
        # Get Apps with good rating
        df_filtered = self.data_frame[(self.data_frame['Category'] != '1.9') & (self.data_frame['Rating'] > 4.0)]
        s_rating = df_filtered.groupby('Rating').agg({'Reviews':'mean'})
        df_rating = pd.DataFrame(s_rating)
        plt.bar(df_rating.index, (df_rating['Reviews'].values/10**6), color='r', width=0.01)
        plt.xlabel('Rating')
        plt.ylabel('No Of Review (millions)')
        plt.savefig('plots/barhighraterevs.png')
        plt.clf()

    def find_lowrev_highrate(self):
        df_filtered = self.data_frame[(self.data_frame['Category'] != '1.9') & (self.data_frame['Rating'] > 4.0)]
        sns_plot = sns.scatterplot(data=df_filtered, x='Rating', y='Price', size='Reviews', legend=False, sizes=(20,2000))
        fig = sns_plot.get_figure()
        # Find that low reviews, high rating could because of low price. Find inference in bubble plot
        fig.savefig('plots/bubblepltpricerevrating.png')
        plt.clf()
