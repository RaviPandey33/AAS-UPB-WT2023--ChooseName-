import cocoex
from pflacco.sampling import create_initial_sample
from pflacco.classical_ela_features import *

class Feature_Extraction:

    
    def __init__(self):
        pass

    
    def gen_features(self,normalized = False):
    
        features = []
        BBOB2 = cocoex.Suite("bbob", f"instances:1-5", f"function_indices:1-24 dimensions:2,3,5,10")

        # Create an empty DataFrame to store the results
        ela_df_columns = ['fid', 'iid', 'dim', 'ela_meta', 'ela_distr', 'ela_level', 'ela_local', 'ela_curv', 'ela_conv']
        ela_df = pd.DataFrame(columns=ela_df_columns)



        for problem in BBOB2:
            dim = problem.dimension
            fid = problem.id_function
            iid = problem.id_instance

            # Create sample
            X = create_initial_sample(dim, lower_bound = -5, upper_bound = 5)
            y = X.apply(lambda x: problem(x), axis = 1)

            if normalized:
                # Normalize y values
                ymin = y.min()
                ymax = y.max()
                y = (y - ymin) / (ymax - ymin)
               

               # Classical ELA features
            ela_meta = calculate_ela_meta(X, y)
            ela_distr = calculate_ela_distribution(X, y)
            ela_level = calculate_ela_level(X, y, ela_level_resample_iterations = 3)

            # Compute the remaining 3 feature sets from the classical ELA features which do require additional function evaluations
            ela_local = calculate_ela_local(X, y, f = problem, dim = dim, lower_bound = -5, upper_bound = 5)
            ela_curv = calculate_ela_curvate(X, y, f = problem, dim = dim, lower_bound = -5, upper_bound = 5)
            ela_conv = calculate_ela_conv(X, y, f = problem)

            #Cell Mapping
            # Compute cell mapping angle feature set from the convential ELA features
            if(dim < 5):
                cm_angle = calculate_cm_angle(X, y, blocks = 3, lower_bound = -5, upper_bound = 5)



            #Dispersion features
            # Compute disp feature set from the convential ELA features
            disp = calculate_dispersion(X, y)

            #IC features
            # Compute ic feature set from the convential ELA features
            ic = calculate_information_content(X, y)

            #NBC features
            nbc = calculate_nbc(X, y)

            #PCA features
            pca = calculate_pca(X,y)



            data = pd.DataFrame({**{'fid': fid}, **{'dim': dim}, **{'iid': iid}, ** ela_meta,  **ela_distr, **ela_level, **ela_local, **ela_curv, **ela_conv, **disp, **ic, **nbc, **pca, **cm_angle}, index = [0])

            
            features.append(data)
        
            
        ela_df = pd.concat(features).reset_index(drop = True)
        print(f"Final row count of DataFrame: {ela_df.shape[0]}")
        if normalized == False:
            file_name = "features.csv"
        else:
            file_name = "n_features.csv"
        ela_df.to_csv(file_name, encoding='utf-8', index=False)
        
        
        if normalized == False:
            file_path = 'features.csv'
        else:
            file_path = 'n_features.csv'
            
        # Load CSV into pandas DataFrame
        df = pd.read_csv(file_path)


        # Extract columns for which you want to calculate the median
        columns_to_calculate_median = ela_df.columns.difference(['fid', 'dim'])
        # print(columns_to_calculate_median)
        
        # Group by fid and dim, calculate the median for selected columns
        result = ela_df.groupby(['fid', 'dim'])[columns_to_calculate_median].median().reset_index()


        # cm_angle columns
        cm_angle_columns = [
            'cm_angle.angle_mean',
            'cm_angle.angle_sd',
            'cm_angle.costs_runtime',
            'cm_angle.dist_ctr2best_mean',
            'cm_angle.dist_ctr2best_sd',
            'cm_angle.dist_ctr2worst_mean',
            'cm_angle.dist_ctr2worst_sd',
            'cm_angle.y_ratio_best2worst_mean',
            'cm_angle.y_ratio_best2worst_sd'
        ]

        # Group by fid, calculate the mean for selected cm_angle columns with dims 2 and 3
        mean_values = result[result['dim'].isin([2, 3])].groupby(['fid'])[cm_angle_columns].mean()

        # Create a new DataFrame with the mean values
        mean_df = pd.DataFrame(mean_values).reset_index()
        mean_df.columns = ['fid'] + [f'{col}_mean' for col in cm_angle_columns]

        # Merge the mean_df with the original result DataFrame on fid
        result = pd.merge(result, mean_df, on='fid', how='left')

        # Update the values for dim 5 and 10 with the calculated means
        for col in cm_angle_columns:
            result.loc[result['dim'].isin([5, 10]), col] = result[f'{col}_mean']

        # Drop the unnecessary columns
        result = result.drop([f'{col}_mean' for col in cm_angle_columns], axis=1)
        # Drop the 'iid' column
        result = result.drop('iid', axis=1)



        file_name = 'n_median_features.csv'
        result.to_csv(file_name, encoding='utf-8', index=False)
        
        return result

fe = Feature_Extraction()
df = fe.gen_features(True)
