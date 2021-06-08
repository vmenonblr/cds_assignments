import matplotlib as mp
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import seaborn as sns

from numpy import arange


class DataVisualiser:
    # Constructor
    def __init__(self, df):
        self.data_frame = df
        plt.rc('figure', figsize=(10, 8))

    # Private methods
    # Removing zero rated apps and an outlier
    def __cleanup_rating(self):
        flt1 = self.data_frame['Rating'] > 0
        flt2 = self.data_frame['Rating'] != 19.0
        df_filtered = self.data_frame.where(flt1 & flt2)
        df_filtered['Rating'].fillna(0)
        return df_filtered

    # Public Methods
    # Plot no of apps against category
    def plot_num_apps_in_category(self):
        try:
            fig = plt.figure()
            # Wanted to try using subplots in a figure
            s_plot = fig.add_subplot(1, 1, 1)
            s_plot.set_ylabel('No Of Apps')
            s_plot.set_title('Categories Vs No Of Apps')
            # Get series group by
            s_cat = self.data_frame.groupby('Category').count()
            # Convert series to data frame
            df_plot = pd.DataFrame(s_cat)
            # Remove the error row
            df_plot = df_plot.drop(['1.9'])
            # Plot bar chart
            s_plot.bar(range(len(df_plot.index)), df_plot['App'].values, color='g')
            # Want the labels in x-axis to be vertically oriented
            s_plot.set_xticks(range(len(df_plot.index)))
            s_plot.set_xticklabels(df_plot.index, rotation='vertical')
            # Render the plot
            s_plot.get_figure().tight_layout()
            # Save it into a png file
            #plt.show()
            plt.savefig("plots/catvsnumapps.png")
            # Have to clear the figure
            plt.clf()
        except RuntimeError as err:
            print("Runtime Error {0}".format(err))
        # Should avoid generic exception handling. Fot the time being it is ok :)
        except:
            print("Unexpected Error {0}".format(sys.exc_info()[0]))

    # Plot the free vs paid for each category
    def plot_dist_free_paid_on_cat(self):
        try:
            s_cat_type = self.data_frame.groupby(['Category', 'Type']).count()
            s_cat_type = s_cat_type.unstack('Type').fillna(0)
            df_plot = pd.DataFrame(s_cat_type)
            # Remove the error row
            df_plot = df_plot.drop(['1.9'])
            l_free_apps_nums = df_plot['App', 'Free'].values
            l_paid_apps_nums = df_plot['App', 'Paid'].values
            # Create the Stacked bar plot based on Free and Paid
            plt.bar(range(len(df_plot.index)), l_free_apps_nums, color='g', label='Free')
            plt.bar(range(len(df_plot.index)), l_paid_apps_nums, bottom=l_free_apps_nums, color='b', label='Paid')
            # Want labels to be vertically oriented
            plt.xticks(range(len(df_plot.index)), df_plot.index, rotation='vertical')
            plt.title("Free Vs Paid apps per Category")
            # Add the label in the plot
            plt.legend()
            #plt.show()
            plt.savefig('plots/stackfreepaidcat.png')
            plt.clf()

        except:
            print("Unexpected Error {0}".format(sys.exc_info()[0]))

    # Scatter plot for ratings for each rating range. Will need binning
    def plot_hist_ratings(self):
        try:
            df_filtered = self.__cleanup_rating()
            # Binning the data frame
            bins = [0, 1, 2, 3, 4, 5]
            df_filtered['binned'] = pd.cut(df_filtered['Rating'], bins, labels=['0-1', '1-2', '2-3', '3-4', '4-5'])
            # Strip Plot based on the bins
            sns_plot = sns.stripplot(x='binned', y='Rating', data=df_filtered)
            fig = sns_plot.get_figure()
            fig.savefig('plots/stripplotforratings.png')
            plt.clf()

        except:
            print("Unexpected Error {0}".format(sys.exc_info()[0]))

    def plot_box_cat_rating(self):
        try:
            df_filtered = self.__cleanup_rating()
            sns_plot = sns.boxplot(x=df_filtered['Rating'], y=df_filtered['Category'])
            fig = sns_plot.get_figure()
            fig.savefig('plots/boxplotcatratings.png')
            plt.clf()
        except:
            print("Unexpected Error {0}".format(sys.exc_info()[0]))

    def plot_bar_cat_installs(self):
        s_cat_installs = self.data_frame.groupby('Category')['Installs'].sum()
        # Aggregate on installs for every category
        df_plot = pd.DataFrame(s_cat_installs)
        df_plot = df_plot.drop(['1.9'])
        plt.bar(range(len(df_plot.index)), df_plot['Installs'].values, color='b')
        plt.xticks(range(len(df_plot.index)), df_plot.index, rotation='vertical')
        plt.title("Installs Vs Category")
       # plt.show()
        plt.savefig('plots/barplotinstallcat.png')
        plt.clf()
