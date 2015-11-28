from dart_model_main import *

#############################################################################################################
## ALL GLOBAL VARIABLES #####################################################################################
#############################################################################################################

major_metro_names = ['Phoenix AZ', 'Houston TX', 'Seattle-Tacoma WA', 'Orlando-Daytona Beach-Melbourne FL', 
                      'Miami-Ft. Lauderdale FL', 'Sacramento-Stockton-Modesto CA', 'Detroit MI', 'Greenville-New Bern-Washington NC', 
                      'Lexington KY', 'Cincinnati OH', 'Monterey-Salinas CA', 
                      'Minneapolis-St. Paul MN', 'Louisville KY', 'San Diego CA', 'Portland-Auburn ME', 
                      'Milwaukee WI', 'Columbus OH', 'Portland OR', 'New York NY', 'San Antonio TX', 'Ft. Myers-Naples FL', 
                      'Colorado Springs-Pueblo CO','Pittsburgh PA', 'San Francisco-Oakland-San Jose CA','Santa Barbara-Santa Maria-San Luis Obispo CA', 
                      'Philadelphia PA', 'Atlanta GA', 'Charlotte NC', 'Oklahoma City OK', 'Albany-Schenectady-Troy NY', 
                      'Tulsa OK', 'Cleveland-Akron (Canton) OH', 'Omaha NE', 'Albuquerque-Santa Fe NM', 
                      'Jacksonville FL', 'Birmingham AL', 'Des Moines-Ames IA',
                      'Buffalo NY', 'Raleigh-Durham (Fayetteville) NC', 'Dallas-Ft. Worth TX', 
                      'Greenville-Spartanburg SC-Asheville NC-Anderson SC', 'West Palm Beach-Ft. Pierce FL', 
                      'Chicago IL', 'Providence RI-New Bedford MA', 'Baltimore MD','Salt Lake City UT', 'Tampa-St. Petersburg (Sarasota) FL', 
                      'Indianapolis IN', 'Rochester NY', 'Florence-Myrtle Beach SC', 'Los Angeles CA', 'Gainesville FL', 'Charleston SC',
                      'Honolulu HI', 'Memphis TN', 'Columbia-Jefferson City MO', 'Las Vegas NV', 'Boston MA-Manchester NH', 
                      'Kansas City MO', 'Dayton OH', 'Harrisburg-Lancaster-Lebanon-York PA', 'New Orleans LA', 'Austin TX', 
                      'Columbia SC', 'Denver CO', 'Anchorage AK', 'Washington DC (Hagerstown MD)', 'St. Louis MO', 
                      'Madison WI', 'Columbus GA']

master_list_filename = 's3n://move-data-engineering-p1/data/reference/users/krux/audience_segment_master.csv'

raw_dart_log_for_predict = 's3n://move-data-engineering-p1/data/events/dart/current/*20150801*'

model_prefix = '/root/dart/DS-Analysis/dart_buyer_model/models/leads_model_v2_'

#############################################################################################################

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        raw_dart_log_for_predict = sys.argv[1]
    
    sparkcontext = initialize_sparkcontext()

    models, jy_and_scores = predict_using_pickled_models(raw_dart_log_for_predict, 
                                                         master_list_filename, 
                                                         major_metro_names, 
                                                         model_prefix, 
                                                         sparkcontext)
    