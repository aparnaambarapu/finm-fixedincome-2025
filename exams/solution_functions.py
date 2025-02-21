import numpy as np

def bond_dirty_price(ytm: float, ttm: float, coupon_rate: float, frequency: int=2, face_value: float=100) -> float:
    """Calculate dirty bond price.

    Parameters
    ----------
    ytm : float
        Yield to maturity.
    ttm : float
        Time to maturity (years).
    coupon_rate : float
        The coupon rate of the bond.
    frequency : int, optional
        The frequency applicable to the coupon rate and the ytm convention, by default 2
    face_value : float, optional
        The face value of the bond, by default 100

    Returns
    -------
    float
        The dirty price of the bond.
    """


    full_periods = int(np.floor(ttm * frequency))  # Full periods to maturity
    stub_period = ttm * frequency - full_periods  # Fractional periods to maturity
    
    cpn_cf = (coupon_rate / frequency) * face_value  # Coupon per period
    r = ytm / frequency  # Periodic yield
    
    round_period_pv = cpn_cf * (1 - (1 + r) ** -full_periods) / r + face_value * (1 + r) ** -full_periods
    
    return (round_period_pv + cpn_cf * (stub_period > 0)) * (1 + r) ** -stub_period

def bond_modified_duration_numerical(ytm: float, ttm: float, coupon_rate: float, frequency: int=2, face_value: float=100, shock: float=0.00001) -> float:
    """Calculate the bond's modified duration with respect to the inputted YTM (including its rate conventions, e.g. not with respect to the continuously compounded rate) using numerical approximation.

    Parameters
    ----------
    ytm : float
        Yield to maturity.
    ttm : float
        Time to maturity (years).
    coupon_rate : float
        The coupon rate of the bond.
    frequency : int, optional
        The frequency applicable to the coupon rate and the ytm convention, by default 2
    face_value : float, optional
        The face value of the bond, by default 100
    shock: float, optional
        The shock to apply to the YTM, by default 0.001

    Returns
    -------
    float
        The dirty price of the bond.
    """
    
    bond_price = bond_dirty_price(ytm=ytm, ttm=ttm, coupon_rate=coupon_rate, frequency=frequency, face_value=face_value)
    up = bond_dirty_price(ytm=ytm + shock, ttm=ttm, coupon_rate=coupon_rate, frequency=frequency, face_value=face_value)
    down = bond_dirty_price(ytm=ytm - shock, ttm=ttm, coupon_rate=coupon_rate, frequency=frequency, face_value=face_value)

    return -(up - down) / (2 * shock) / bond_price

def bond_modified_convexity_numerical(ytm: float, ttm: float, coupon_rate: float, frequency: int=2, face_value: float=100, shock: float=0.00001) -> float:
    """Calculate the bond's modified durationconvexity with respect to the inputted YTM (including its rate conventions, e.g. not with respect to the continuously compounded rate) using numerical approximation.
    This is the second derivative of the bond price with respect to the YTM, divided by the bond price.

    Parameters
    ----------
    ytm : float
        Yield to maturity.
    ttm : float
        Time to maturity (years).
    coupon_rate : float
        The coupon rate of the bond.
    frequency : int, optional
        The frequency applicable to the coupon rate and the ytm convention, by default 2
    face_value : float, optional
        The face value of the bond, by default 100
    shock: float, optional
        The shock to apply to the YTM, by default 0.001

    Returns
    -------
    float
        The dirty price of the bond.
    """
    
    bond_price = bond_dirty_price(ytm=ytm, ttm=ttm, coupon_rate=coupon_rate, frequency=frequency, face_value=face_value)

    mod_dur_up = bond_modified_duration_numerical(ytm=ytm + shock, ttm=ttm, coupon_rate=coupon_rate, frequency=frequency, face_value=face_value, shock=shock)
    mod_dur_down = bond_modified_duration_numerical(ytm=ytm - shock, ttm=ttm, coupon_rate=coupon_rate, frequency=frequency, face_value=face_value, shock=shock)

    bond_price_up = bond_dirty_price(ytm=ytm + shock, ttm=ttm, coupon_rate=coupon_rate, frequency=frequency, face_value=face_value)
    bond_price_down = bond_dirty_price(ytm=ytm - shock, ttm=ttm, coupon_rate=coupon_rate, frequency=frequency, face_value=face_value)

    up = mod_dur_up * bond_price_up
    down = mod_dur_down * bond_price_down

    return -(up - down) / (2 * shock) / bond_price