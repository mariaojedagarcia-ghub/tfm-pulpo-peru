# --- LÓGICA DE PREDICCIÓN ---
    # Tu modelo espera 13 columnas. Según tus datos, el orden más probable es:
    # [año, mes, nino12, soi, nino12_lag1, soi_lag1, nino12_lag2, soi_lag2, nino12_lag3, soi_lag3, + 3 extras]
    
    import datetime
    hoy = datetime.datetime.now()
    
    # Creamos la lista de 13 valores
    inputs = [[
        hoy.year,          # 1. Año
        hoy.month,         # 2. Mes
        nino_input,        # 3. nino12
        soi_input,         # 4. soi
        nino_input,        # 5. nino12_lag1 (simulado)
        soi_input,         # 6. soi_lag1 (simulado)
        nino_input,        # 7. nino12_lag2 (simulado)
        soi_input,         # 8. soi_lag2 (simulado)
        nino_input,        # 9. nino12_lag3 (simulado)
        soi_input,         # 10. soi_lag3 (simulado)
        0, 0, 0            # 11, 12, 13. Relleno para ONI/Otros
    ]]
    
    prediccion_raw = model.predict(inputs)[0]
    prediccion_final = max(0, float(prediccion_raw))
