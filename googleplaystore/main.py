import loaddata
import plotvisualisations
import datainsights

def init_processing():
    # Load Data
    data_frame = loaddata.load_data("./data/googleplaystore.csv")
    print(data_frame)
    return data_frame

def plot_chats(df):
    visual = plotvisualisations.DataVisualiser(df)
    visual.plot_num_apps_in_category()
    visual.plot_dist_free_paid_on_cat()
    visual.plot_hist_ratings()
    visual.plot_box_cat_rating()
    visual.plot_bar_cat_installs()

def plot_insights(df):
    insights = datainsights.DataInsigthFinder(df)
    insights.plot_scatter_size_price()
    insights.plot_scatter_rating_installs_cat()
    insights.plot_count_cat_year_apps()
    insights.find_highest_paid_app()
    insights.find_top_rated_app_geniuine()
    insights.find_lowrev_highrate()


if __name__ == '__main__':
    df = init_processing()
    plot_chats(df)
    plot_insights(df)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
