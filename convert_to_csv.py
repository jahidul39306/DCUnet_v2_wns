import pandas as pd
def save_to_csv(output_path, collected):
    rows = []
    for (model_name, ts_name), metrics in collected.items():
        row = {
            "Model": model_name,
            "Dataset_TS": ts_name,
            
            # Noisy Metrics
            "Noisy_STOI": metrics["noisy"].get("STOI"),
            "Noisy_SI-SDR": metrics["noisy"].get("SI-SDR"),
            "Noisy_PESQ": metrics["noisy"].get("PESQ"),
            
            # Enhanced Metrics
            "Enhanced_STOI": metrics["enhanced"].get("STOI"),
            "Enhanced_SI-SDR": metrics["enhanced"].get("SI-SDR"),
            "Enhanced_PESQ": metrics["enhanced"].get("PESQ"),
            
            # Normalized Enhanced Metrics
            "Norm_STOI": metrics["enhanced_norm"].get("STOI-norm"),
            "Norm_SI-SDR": metrics["enhanced_norm"].get("SI-SDR-norm"),
            "Norm_PESQ": metrics["enhanced_norm"].get("PESQ-norm"),
        }
        rows.append(row)

    # Convert to DataFrame and export to CSV
    df = pd.DataFrame(rows)
    df.to_csv(output_path+"/speech_enhancement_results.csv", index=False)
    print("Saved to speech_enhancement_results.csv")