import pickle
import json
if __name__ == "__main__":
    # Load dict_negative_audios
    with open("nami_record/dict_negative_audios_newlstm.pkl", "rb") as f:
        dict_negative_audios = pickle.load(f)
    
    dic_result = {"rule1": {}, "rule2": {}, "rule3": {}, "rule4": {}}


    # Rule 1
    # Threshold to compare
    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        # Keys are filenames
        dic_result["rule1"][threshold] = {"lstm": {"false_alarm": None, "miss_rate": None}, "dscnn": {"false_alarm": None, "miss_rate": None}}
        lstm_fa_count = 0
        dscnn_fa_count = 0
        for key in dict_negative_audios:
            scores = dict_negative_audios[key]
            for score in scores:
                lstm_pos_score = score[0]
                dscnn_pos_score = score[2]
                
                # LSTM 
                if lstm_pos_score > threshold:
                    lstm_fa_count += 1
                
                # DSCNN
                if dscnn_pos_score > threshold:
                    dscnn_fa_count += 1
        
        dic_result["rule1"][threshold]["lstm"]["false_alarm"] = lstm_fa_count
        dic_result["rule1"][threshold]["dscnn"]["false_alarm"] = dscnn_fa_count
    
    # Rule 2
    # Threshold to compare
    for threshold in [-0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        # Keys are filenames
        dic_result["rule2"][threshold] = {"lstm": {"false_alarm": None, "miss_rate": None}, "dscnn": {"false_alarm": None, "miss_rate": None}}
        lstm_fa_count = 0
        dscnn_fa_count = 0
        for key in dict_negative_audios:
            scores = dict_negative_audios[key]
            for score in scores:
                lstm_pos_score = score[0]
                lstm_neg_score = score[1]
                dscnn_pos_score = score[2]
                dscnn_neg_score = score[3]

                # LSTM
                # print(lstm_pos_score - lstm_neg_score)
                # print("-----------------------------------------")
                # print(lstm_pos_score)
                # print(lstm_neg_score)
                # print("-----------------------------------------")
                if (lstm_pos_score - lstm_neg_score) > threshold:
                    lstm_fa_count += 1
                
                # DSCNN
                if (dscnn_pos_score - dscnn_neg_score) > threshold:
                    dscnn_fa_count += 1

        dic_result["rule2"][threshold]["lstm"]["false_alarm"] = lstm_fa_count
        dic_result["rule2"][threshold]["dscnn"]["false_alarm"] = dscnn_fa_count
    

    # Rule 3
    # Threshold to compare
    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        # Keys are filenames
        dic_result["rule3"][threshold] = {"false_alarm": None, "miss_rate": None}
        fa_count = 0
        for key in dict_negative_audios:
            scores = dict_negative_audios[key]
            for score in scores:
                lstm_pos_score = score[0]
                dscnn_pos_score = score[2]
                # LSTM & DSCNN
                if (lstm_pos_score > threshold) and (dscnn_pos_score > 0.6):
                    fa_count += 1

        dic_result["rule3"][threshold]["false_alarm"] = fa_count

    
    # Rule 4
    for threshold in [-0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    # Keys are filenames
        dic_result["rule4"][threshold] = {"false_alarm": None, "miss_rate": None}
        fa_count = 0
        for key in dict_negative_audios:
            scores = dict_negative_audios[key]
            for score in scores:
                lstm_pos_score = score[0]
                lstm_neg_score = score[1]
                dscnn_pos_score = score[2]
                dscnn_neg_score = score[3]
                # LSTM & DSCNN
                if ((lstm_pos_score - lstm_neg_score) > threshold) and ((dscnn_pos_score - dscnn_neg_score) > 0.6):
                    fa_count += 1

        dic_result["rule4"][threshold]["false_alarm"] = fa_count

    with open("nami_record/result_testset_easy.json", "w") as f:
        json.dump(dic_result, f)
    
    print(dic_result)