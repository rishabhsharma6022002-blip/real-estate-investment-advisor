def predict_future_price(current_price, annual_growth_rate=0.08, years=5):

    future_price = current_price * (
        1 + annual_growth_rate
    ) ** years

    return future_price


