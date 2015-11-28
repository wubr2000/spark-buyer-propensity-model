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
# dart_raw_logs = 's3n://move-data-engineering-p1/data/events/dart/current/NetworkImpressions_288077_20150801_07.gz'
dart_raw_logs = 's3n://move-data-engineering-p1/data/events/dart/current/*20150801*'
model_prefix = '/root/dart/DS-Analysis/dart_buyer_model/models/leads_model_v2_'

#############################################################################################################


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        dart_raw_logs = sys.argv[1]
    
    sparkcontext = initialize_sparkcontext()
    
    all_aud_id = aud_segment_id_master_list(master_list_filename, sparkcontext)
    dart_has_audience_id, dart_headers, metro_id_name_dict = process_raw_log_into_rdd(dart_raw_logs, major_metro_names, sparkcontext)

    jy_sparse_vector_rdd, leads_labels_rdd = build_sparse_vector_and_leads_labels_rdd(all_aud_id, dart_has_audience_id, dart_headers)

    nonsampled_labeledpoints_with_metroid = build_nonsampled_labeledpoints(leads_labels_rdd, jy_sparse_vector_rdd)
    upsampled_labeledpoints_with_metroid = build_upsampled_labeledpoints(leads_labels_rdd, jy_sparse_vector_rdd)

    train_and_save_logistic_models(upsampled_labeledpoints_with_metroid, all_aud_id, metro_id_name_dict, model_prefix)