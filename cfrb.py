# Wykres alokacji w czasie
        if auto_rebalancing:
            st.subheader("🎯 Precyzja AUTO-CASH-FLOW REBALANCING")
            st.info("W trybie AUTO alokacja powinna być idealnie utrzymana na poziomie docelowym")
        elif band_rebalancing:
            st.subheader("📊 BAND REBALANCING - Pasma tolerancji")
            st.info(f"Rebalansing następuje gdy metal wyjdzie poza swoje pasmo • Liczba interwencji: {len(triggers)}")
        else:
            st.subheader("📊 Zmiana alokacji w czasie")
        
        fig_allocation = go.Figure()
        
        # Dodaj pasma tolerancji dla BAND REBALANCING
        if band_rebalancing and bands:
            metals_info = [
                ('Złoto', target_gold, bands['gold'], 'gold'),
                ('Srebro', target_silver, bands['silver'], 'silver'), 
                ('Platyna', target_platinum, bands['platinum'], 'lightgray'),
                ('Pallad', target_palladium, bands['palladium'], 'lightsteelblue')
            ]
            
            for metal_name, target_pct, band_width, color in metals_info:
                # Górna granica pasma
                fig_allocation.add_hline(
                    y=target_pct + band_width, 
                    line_dash="dash", 
                    line_color=color,
                    opacity=0.7,
                    annotation_text=f"{metal_name} +{band_width}%",
                    annotation_position="right"
                )
                # Dolna granica pasma
                fig_allocation.add_hline(
                    y=target_pct - band_width, 
                    line_dash="dash", 
                    line_color=color,
                    opacity=0.7,
                    annotation_text=f"{metal_name} -{band_width}%",
                    annotation_position="right"
                )
                # Cel (środek pasma)
                fig_allocation.add_hline(
                    y=target_pct, 
                    line_dash="solid", 
                    line_color=color,
                    opacity=0.5,
                    line_width=1
                )
        elif not band_rebalancing:
            # Dodaj linie docelowych alokacji jako punkty odniesienia (dla innych trybów)
            for metal, target_pct in [('Złoto', target_gold), ('Srebro', target_silver), 
                                     ('Platyna', target_platinum), ('Pallad', target_palladium)]:
                fig_allocation.add_hline(
                    y=target_pct, 
                    line_dash="dash", 
                    line_color="red",
                    annotation_text=f"Cel {metal}: {target_pct}%",
                    annotation_position="right"
                )
        
        # Dodaj rzeczywiste alokacje
        colors_allocation = {'Złoto_%': 'gold', 'Srebro_%': 'silver', 'Platyna_%': 'lightgray', import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
import random

# Konfiguracja strony
st.set_page_config(
    page_title="Symulator Inwestycji w Metale Szlachetne - LBMA",
    page_icon="🥇",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Stałe przeliczeniowe
TROY_OUNCE_TO_GRAMS = 31.1035  # 1 uncja trojańska = 31.1035 gramów

@st.cache_data
def load_lbma_data():
    """Wczytaj i przygotuj dane LBMA"""
    try:
        # Wczytanie danych z pliku CSV
        df = pd.read_csv('lbma_data.csv')
        
        # Konwersja dat - upewnij się, że są w formacie datetime
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
        # Usuń rekordy z nieprawidłowymi datami
        df = df.dropna(subset=['Date'])
        
        # Konwersja z uncji na gramy (podzielenie przez wagę uncji trojańskiej)
        df['Gold_EUR_per_gram'] = df['Gold_EUR'] / TROY_OUNCE_TO_GRAMS
        df['Silver_EUR_per_gram'] = df['Silver_EUR'] / TROY_OUNCE_TO_GRAMS
        df['Platinum_EUR_per_gram'] = df['Platinum_EUR'] / TROY_OUNCE_TO_GRAMS
        df['Palladium_EUR_per_gram'] = df['Palladium_EUR'] / TROY_OUNCE_TO_GRAMS
        
        # Posortowanie według daty
        df = df.sort_values('Date').reset_index(drop=True)
        
        return df
    except FileNotFoundError:
        st.error("❌ Nie znaleziono pliku lbma_data.csv. Upewnij się, że plik znajduje się w tym samym folderze co aplikacja.")
        return None
    except Exception as e:
        st.error(f"❌ Błąd podczas wczytywania danych: {str(e)}")
        return None

def calculate_monthly_returns(df, start_date, end_date, months):
    """Oblicz miesięczne zmiany cen na podstawie rzeczywistych danych LBMA"""
    
    try:
        # Konwersja dat na datetime
        if isinstance(start_date, str):
            start_date_dt = pd.to_datetime(start_date)
        elif hasattr(start_date, 'date'):
            start_date_dt = pd.to_datetime(start_date)
        else:
            start_date_dt = pd.to_datetime(str(start_date))
            
        if isinstance(end_date, str):
            end_date_dt = pd.to_datetime(end_date)
        elif hasattr(end_date, 'date'):
            end_date_dt = pd.to_datetime(end_date)
        else:
            end_date_dt = pd.to_datetime(str(end_date))
        
        # Upewnij się, że kolumna Date jest w formacie datetime
        if not pd.api.types.is_datetime64_any_dtype(df['Date']):
            df = df.copy()
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.dropna(subset=['Date'])
        
        # Usuń timezone jeśli istnieje
        if hasattr(start_date_dt, 'tz') and start_date_dt.tz is not None:
            start_date_dt = start_date_dt.tz_localize(None)
        if hasattr(end_date_dt, 'tz') and end_date_dt.tz is not None:
            end_date_dt = end_date_dt.tz_localize(None)
            
        if hasattr(df['Date'].dtype, 'tz') and df['Date'].dtype.tz is not None:
            df = df.copy()
            df['Date'] = df['Date'].dt.tz_localize(None)
        
        # Filtrowanie danych w wybranym okresie
        df_filtered = df[(df['Date'] >= start_date_dt) & (df['Date'] <= end_date_dt)].copy()
        
    except Exception as e:
        st.error(f"Błąd w calculate_monthly_returns: {str(e)}")
        return generate_simulated_returns(months), pd.DataFrame()
    
    if len(df_filtered) == 0:
        st.warning(f"⚠️ Brak danych od {start_date.strftime('%Y-%m-%d')}. Używam najnowszych dostępnych danych.")
        df_filtered = df.tail(months * 30).copy()  # Przybliżenie: 30 dni = 1 miesiąc
    
    # Grupowanie po miesiącach i obliczanie średnich cen
    df_filtered['YearMonth'] = df_filtered['Date'].dt.to_period('M')
    monthly_avg = df_filtered.groupby('YearMonth')[
        ['Gold_EUR_per_gram', 'Silver_EUR_per_gram', 'Platinum_EUR_per_gram', 'Palladium_EUR_per_gram']
    ].mean().reset_index()
    
    if len(monthly_avg) < 2:
        st.warning("⚠️ Za mało danych historycznych. Używam symulowanych zmian.")
        return generate_simulated_returns(months)
    
    # Obliczenie miesięcznych zwrotów
    returns = {}
    metals = ['Gold', 'Silver', 'Platinum', 'Palladium']
    
    for metal in metals:
        col_name = f'{metal}_EUR_per_gram'
        if col_name in monthly_avg.columns:
            prices = monthly_avg[col_name].values
            monthly_changes = []
            
            for i in range(1, len(prices)):
                change = (prices[i] - prices[i-1]) / prices[i-1]
                monthly_changes.append(change)
            
            # Jeśli potrzebujemy więcej danych niż mamy, uzupełniamy losowo z historycznych
            while len(monthly_changes) < months:
                monthly_changes.extend(monthly_changes[:min(len(monthly_changes), months - len(monthly_changes))])
            
            returns[metal.lower()] = monthly_changes[:months]
    
    return returns, monthly_avg

def generate_simulated_returns(months):
    """Generuj symulowane zwroty jako fallback"""
    np.random.seed(42)  # Dla powtarzalności
    return {
        'gold': np.random.normal(0.008, 0.025, months).tolist(),
        'silver': np.random.normal(0.005, 0.035, months).tolist(),
        'platinum': np.random.normal(0.003, 0.030, months).tolist(),
        'palladium': np.random.normal(0.010, 0.045, months).tolist()
    }

def get_current_prices(df):
    """Pobierz najnowsze ceny metali za gram"""
    if df is None or len(df) == 0:
        return None
    
    latest_data = df.iloc[-1]
    return {
        'gold': latest_data['Gold_EUR_per_gram'],
        'silver': latest_data['Silver_EUR_per_gram'],
        'platinum': latest_data['Platinum_EUR_per_gram'],
        'palladium': latest_data['Palladium_EUR_per_gram'],
        'date': latest_data['Date'].strftime('%Y-%m-%d')
    }

# Tytuł aplikacji
st.title("🥇 Symulator Inwestycji w Metale Szlachetne")
st.markdown("### Oparte na rzeczywistych danych LBMA (ceny za gram w EUR)")

# Wczytanie danych
with st.spinner('📊 Wczytywanie danych LBMA...'):
    lbma_df = load_lbma_data()

if lbma_df is not None:
    current_prices = get_current_prices(lbma_df)
    
    # Wyświetlenie aktualnych cen
    if current_prices:
        st.sidebar.markdown("### 💰 Aktualne ceny (EUR/gram)")
        st.sidebar.markdown(f"**Data:** {current_prices['date']}")
        st.sidebar.markdown(f"**🥇 Złoto:** {current_prices['gold']:.2f} €/g")
        st.sidebar.markdown(f"**🥈 Srebro:** {current_prices['silver']:.3f} €/g")
        st.sidebar.markdown(f"**⚪ Platyna:** {current_prices['platinum']:.2f} €/g")
        st.sidebar.markdown(f"**⚫ Pallad:** {current_prices['palladium']:.2f} €/g")
        st.sidebar.markdown("---")

# Sidebar - parametry symulacji
st.sidebar.header("⚙️ Parametry Symulacji")

initial_investment = st.sidebar.number_input(
    "Kapitał początkowy (EUR)", 
    min_value=100, 
    max_value=1000000, 
    value=10000, 
    step=500
)

monthly_contribution = st.sidebar.number_input(
    "Miesięczna wpłata (EUR)", 
    min_value=0, 
    max_value=10000, 
    value=500, 
    step=50
)

st.sidebar.subheader("⚖️ Konfiguracja rebalansingu")

rebalancing_mode = st.sidebar.selectbox(
    "Strategia rebalansingu",
    options=["Budżet stały", "AUTO-CASH-FLOW", "Pasma tolerancji (BAND)"],
    index=0,
    help="Budżet stały: określasz maksymalną kwotę na rebalansing\nAUTO-CASH-FLOW: system automatycznie dodaje środki potrzebne do idealnego rebalansingu\nPasma tolerancji: rebalansing tylko gdy metal wyjdzie poza swoje pasmo"
)

if rebalancing_mode == "Budżet stały":
    rebalancing_budget = st.sidebar.number_input(
        "Budżet na rebalansing (EUR/miesiąc)", 
        min_value=0, 
        max_value=5000, 
        value=100, 
        step=25,
        help="Dodatkowe środki przeznaczone wyłącznie na rebalansing portfela"
    )
    auto_rebalancing = False
    band_rebalancing = False
    
elif rebalancing_mode == "AUTO-CASH-FLOW":
    st.sidebar.info("🤖 **AUTO-CASH-FLOW REBALANCING**\n\nSystem automatycznie doliczy środki potrzebne do idealnego utrzymania proporcji portfela")
    rebalancing_budget = 0  # Nie używany w trybie auto
    auto_rebalancing = True
    band_rebalancing = False
    
else:  # Pasma tolerancji
    st.sidebar.info("📊 **BAND REBALANCING**\n\nRebalansing tylko gdy metal wyjdzie poza swoje pasmo tolerancji")
    
    # Budżet dla rebalansingu pasma
    rebalancing_budget = st.sidebar.number_input(
        "Budżet na rebalansing (EUR/miesiąc)", 
        min_value=0, 
        max_value=5000, 
        value=200, 
        step=25,
        help="Środki używane gdy metal wyjdzie poza pasmo tolerancji"
    )
    
    st.sidebar.markdown("**Szerokość pasm tolerancji:**")
    
    # Pasma tolerancji dla każdego metalu
    gold_band = st.sidebar.slider(
        "🥇 Złoto - szerokość pasma (%)", 
        min_value=1, max_value=15, value=5, step=1,
        help=f"Pasmo: {target_gold-5}% - {target_gold+5}% (cel: {target_gold}%)"
    )
    
    silver_band = st.sidebar.slider(
        "🥈 Srebro - szerokość pasma (%)", 
        min_value=1, max_value=15, value=7, step=1,
        help=f"Pasmo: {target_silver-7}% - {target_silver+7}% (cel: {target_silver}%)"
    )
    
    platinum_band = st.sidebar.slider(
        "⚪ Platyna - szerokość pasma (%)", 
        min_value=1, max_value=15, value=6, step=1,
        help=f"Pasmo: {target_platinum-6}% - {target_platinum+6}% (cel: {target_platinum}%)"
    )
    
    palladium_band = st.sidebar.slider(
        "⚫ Pallad - szerokość pasma (%)", 
        min_value=1, max_value=15, value=8, step=1,
        help=f"Pasmo: {target_palladium-8}% - {target_palladium+8}% (cel: {target_palladium}%)"
    )
    
    auto_rebalancing = False
    band_rebalancing = True

st.sidebar.markdown("---")

# Wybór okresu inwestycji z kalendarza
if lbma_df is not None and len(lbma_df) > 0:
    try:
        min_date = lbma_df['Date'].min().date()
        max_date = lbma_df['Date'].max().date()
        
        st.sidebar.subheader("📅 Okres inwestycji")
        st.sidebar.markdown(f"*Dostępne dane: {min_date} - {max_date}*")
        
        # Domyślny przedział: ostatnie 2 lata
        default_start = max_date - timedelta(days=365*2)
        default_end = max_date - timedelta(days=30)  # miesiąc przed końcem danych
        
        start_date = st.sidebar.date_input(
            "📅 Data rozpoczęcia inwestycji",
            value=default_start,
            min_value=min_date,
            max_value=max_date - timedelta(days=30)  # przynajmniej miesiąc przed końcem
        )
        
        end_date = st.sidebar.date_input(
            "🏁 Data zakończenia inwestycji",
            value=default_end,
            min_value=start_date + timedelta(days=30) if start_date else min_date,
            max_value=max_date
        )
        
        # Oblicz liczbę miesięcy między datami
        if start_date and end_date and end_date > start_date:
            simulation_months = max(1, int((end_date - start_date).days / 30.44))  # średnio 30.44 dni w miesiącu
            st.sidebar.info(f"📊 Okres symulacji: **{simulation_months} miesięcy** ({(end_date - start_date).days} dni)")
        else:
            simulation_months = 1
            st.sidebar.warning("⚠️ Data zakończenia musi być późniejsza niż rozpoczęcia")
            
    except Exception as e:
        st.sidebar.error(f"Błąd z datami: {str(e)}")
        start_date = None
        end_date = None
        simulation_months = 24
else:
    start_date = None
    end_date = None
    simulation_months = 24

rebalance_frequency = st.sidebar.selectbox(
    "Częstotliwość rebalansingu",
    options=[1, 3, 6, 12],
    index=1,
    format_func=lambda x: f"Co {x} miesiąc(e/y)"
)

st.sidebar.subheader("🎯 Docelowa alokacja (%)")

# Docelowa alokacja
target_gold = st.sidebar.slider("🥇 Złoto (%)", 0, 100, 40, 5)
target_silver = st.sidebar.slider("🥈 Srebro (%)", 0, 100, 30, 5)
target_platinum = st.sidebar.slider("⚪ Platyna (%)", 0, 100, 20, 5)
target_palladium = st.sidebar.slider("⚫ Pallad (%)", 0, 100, 10, 5)

# Sprawdzenie czy suma wynosi 100%
total_allocation = target_gold + target_silver + target_platinum + target_palladium
if total_allocation != 100:
    st.sidebar.error(f"⚠️ Suma alokacji: {total_allocation}%. Musi wynosić 100%!")

# Przycisk uruchomienia symulacji
run_simulation = st.sidebar.button("🚀 Uruchom Symulację", type="primary")

def run_portfolio_simulation(initial_inv, monthly_cont, rebalancing_budget, months, rebalance_freq, allocations, price_changes, auto_rebalancing=False, band_rebalancing=False, bands=None):
    """Główna funkcja symulacji portfela z rzeczywistymi danymi i różnymi strategiami rebalancingu"""
    
    # Inicjalizacja portfela w gramach
    if current_prices:
        initial_grams = {
            'gold': (initial_inv * allocations['gold'] / 100) / current_prices['gold'],
            'silver': (initial_inv * allocations['silver'] / 100) / current_prices['silver'],
            'platinum': (initial_inv * allocations['platinum'] / 100) / current_prices['platinum'],
            'palladium': (initial_inv * allocations['palladium'] / 100) / current_prices['palladium']
        }
    else:
        # Fallback - symulowane ceny startowe
        initial_grams = {
            'gold': (initial_inv * allocations['gold'] / 100) / 75,
            'silver': (initial_inv * allocations['silver'] / 100) / 0.95,
            'platinum': (initial_inv * allocations['platinum'] / 100) / 27,
            'palladium': (initial_inv * allocations['palladium'] / 100) / 26
        }
    
    # Symulowane ceny startowe
    prices = current_prices.copy() if current_prices else {
        'gold': 75.0, 'silver': 0.95, 'platinum': 27.0, 'palladium': 26.0
    }
    
    simulation_data = []
    portfolio_grams = initial_grams.copy()
    total_regular_invested = initial_inv
    total_rebalancing_spent = 0
    rebalancing_triggers = []  # Do śledzenia kiedy nastąpił rebalansing
    
    for month in range(months + 1):
        # Zastosuj zmiany cen (oprócz pierwszego miesiąca)
        if month > 0 and month <= len(price_changes['gold']):
            for metal in prices.keys():
                if metal in price_changes and (month-1) < len(price_changes[metal]):
                    change = price_changes[metal][month-1]
                    prices[metal] *= (1 + change)
            
            # Dodaj regularne miesięczne wpłaty (proporcjonalnie do docelowej alokacji)
            if monthly_cont > 0:
                total_regular_invested += monthly_cont
                for metal in portfolio_grams.keys():
                    target_percent = allocations[metal]
                    eur_to_invest = monthly_cont * (target_percent / 100)
                    grams_to_add = eur_to_invest / prices[metal]
                    portfolio_grams[metal] += grams_to_add
            
            # Sprawdź czy potrzebny jest rebalansing
            rebalancing_needed = False
            rebalancing_spent_this_cycle = 0
            
            # Oblicz aktualną wartość portfela i alokację
            current_values = {
                metal: portfolio_grams[metal] * prices[metal] 
                for metal in portfolio_grams.keys()
            }
            current_total = sum(current_values.values())
            
            # Oblicz różnice od docelowej alokacji
            allocation_differences = {}
            current_allocations = {}
            for metal in portfolio_grams.keys():
                current_percent = (current_values[metal] / current_total) * 100
                target_percent = allocations[metal]
                difference = target_percent - current_percent
                allocation_differences[metal] = difference
                current_allocations[metal] = current_percent
            
            # Różne strategie rebalancingu
            if month % rebalance_freq == 0:
                
                if band_rebalancing and bands:
                    # BAND REBALANCING - sprawdź czy któryś metal wyszedł poza pasmo
                    metals_outside_bands = {}
                    
                    for metal in portfolio_grams.keys():
                        current_pct = current_allocations[metal]
                        target_pct = allocations[metal]
                        band_width = bands[metal]
                        
                        lower_bound = target_pct - band_width
                        upper_bound = target_pct + band_width
                        
                        if current_pct < lower_bound:
                            # Metal poniżej dolnej granicy - trzeba dokupić
                            needed_pct = target_pct - current_pct
                            metals_outside_bands[metal] = needed_pct
                            rebalancing_needed = True
                        elif current_pct > upper_bound:
                            # Metal powyżej górnej granicy - należałoby sprzedać, ale tego nie robimy w cash-flow
                            # Zamiast tego zmniejszamy zakupy innych metali w następnych miesiącach
                            pass
                    
                    if metals_outside_bands and rebalancing_budget > 0:
                        available_budget = rebalancing_budget * rebalance_freq
                        total_needed_pct = sum(metals_outside_bands.values())
                        
                        for metal, needed_pct in metals_outside_bands.items():
                            if rebalancing_spent_this_cycle >= available_budget:
                                break
                                
                            # Proporcjonalne rozdzielenie budżetu
                            budget_ratio = needed_pct / total_needed_pct if total_needed_pct > 0 else 0
                            rebalancing_eur = min(
                                available_budget * budget_ratio,
                                available_budget - rebalancing_spent_this_cycle
                            )
                            
                            if rebalancing_eur > 0:
                                grams_to_add = rebalancing_eur / prices[metal]
                                portfolio_grams[metal] += grams_to_add
                                rebalancing_spent_this_cycle += rebalancing_eur
                        
                        if rebalancing_spent_this_cycle > 0:
                            rebalancing_triggers.append({
                                'month': month,
                                'reason': 'Band violation',
                                'metals': list(metals_outside_bands.keys()),
                                'spent': rebalancing_spent_this_cycle
                            })
                
                elif auto_rebalancing:
                    # AUTO-CASH-FLOW: dodaj tyle środków ile potrzeba dla idealnego rebalansingu
                    underweight_metals = {k: v for k, v in allocation_differences.items() if v > 0}
                    
                    if underweight_metals:
                        for metal, deficit in underweight_metals.items():
                            target_value = current_total * (allocations[metal] / 100)
                            current_value = current_values[metal]
                            needed_eur = target_value - current_value
                            
                            if needed_eur > 0:
                                grams_to_add = needed_eur / prices[metal]
                                portfolio_grams[metal] += grams_to_add
                                rebalancing_spent_this_cycle += needed_eur
                        
                        if rebalancing_spent_this_cycle > 0:
                            rebalancing_triggers.append({
                                'month': month,
                                'reason': 'Auto rebalancing',
                                'metals': list(underweight_metals.keys()),
                                'spent': rebalancing_spent_this_cycle
                            })
                
                else:
                    # Rebalansing z ograniczonym budżetem (poprzednia logika)
                    underweight_metals = {k: v for k, v in allocation_differences.items() if v > 0}
                    
                    if underweight_metals and rebalancing_budget > 0:
                        available_rebalancing_budget = rebalancing_budget * rebalance_freq
                        total_deficit = sum(underweight_metals.values())
                        
                        for metal, deficit in underweight_metals.items():
                            if rebalancing_spent_this_cycle >= available_rebalancing_budget:
                                break
                                
                            deficit_ratio = deficit / total_deficit
                            rebalancing_eur = min(
                                available_rebalancing_budget * deficit_ratio,
                                available_rebalancing_budget - rebalancing_spent_this_cycle
                            )
                            
                            if rebalancing_eur > 0:
                                grams_to_add = rebalancing_eur / prices[metal]
                                portfolio_grams[metal] += grams_to_add
                                rebalancing_spent_this_cycle += rebalancing_eur
                        
                        if rebalancing_spent_this_cycle > 0:
                            rebalancing_triggers.append({
                                'month': month,
                                'reason': 'Fixed budget',
                                'metals': list(underweight_metals.keys()),
                                'spent': rebalancing_spent_this_cycle
                            })
                
                total_rebalancing_spent += rebalancing_spent_this_cycle
        
        # Oblicz wartości w EUR
        current_values = {
            metal: portfolio_grams[metal] * prices[metal] 
            for metal in portfolio_grams.keys()
        }
        total_value = sum(current_values.values())
        
        # Zapisz dane miesiąca
        month_data = {
            'Miesiąc': month,
            'Łączna_wartość': total_value,
            'Złoto_EUR': current_values['gold'],
            'Srebro_EUR': current_values['silver'],
            'Platyna_EUR': current_values['platinum'],
            'Pallad_EUR': current_values['palladium'],
            'Złoto_gramy': portfolio_grams['gold'],
            'Srebro_gramy': portfolio_grams['silver'],
            'Platyna_gramy': portfolio_grams['platinum'],
            'Pallad_gramy': portfolio_grams['palladium'],
            'Cena_złoto': prices['gold'],
            'Cena_srebro': prices['silver'],
            'Cena_platyna': prices['platinum'],
            'Cena_pallad': prices['palladium'],
            'Złoto_%': (current_values['gold'] / total_value) * 100,
            'Srebro_%': (current_values['silver'] / total_value) * 100,
            'Platyna_%': (current_values['platinum'] / total_value) * 100,
            'Pallad_%': (current_values['palladium'] / total_value) * 100,
            'Wpłaty_regularne': total_regular_invested,
            'Budżet_rebalancing': total_rebalancing_spent
        }
        simulation_data.append(month_data)
    
    return pd.DataFrame(simulation_data), portfolio_grams, prices, total_regular_invested, total_rebalancing_spent, rebalancing_triggers

# Uruchomienie symulacji
if run_simulation and total_allocation == 100 and lbma_df is not None and start_date is not None and end_date is not None and end_date > start_date:
    with st.spinner('🔄 Analizowanie danych LBMA i symulowanie inwestycji...'):
        
        target_allocations = {
            'gold': target_gold,
            'silver': target_silver,
            'platinum': target_platinum,
            'palladium': target_palladium
        }
        
        # Oblicz miesięczne zmiany cen na podstawie rzeczywistych danych
        price_changes, monthly_avg = calculate_monthly_returns(lbma_df, start_date, end_date, simulation_months)
        
        # Przygotuj parametry dla BAND REBALANCING
        bands = None
        if band_rebalancing:
            bands = {
                'gold': gold_band,
                'silver': silver_band, 
                'platinum': platinum_band,
                'palladium': palladium_band
            }
        
        # Uruchom symulację
        df, final_grams, final_prices, total_regular, total_rebalancing, triggers = run_portfolio_simulation(
            initial_investment,
            monthly_contribution,
            rebalancing_budget,
            simulation_months,
            rebalance_frequency,
            target_allocations,
            price_changes,
            auto_rebalancing,
            band_rebalancing,
            bands
        )
        
        # Obliczenia finansowe
        total_invested = total_regular + total_rebalancing
        final_value = df['Łączna_wartość'].iloc[-1]
        total_return = final_value - total_invested
        return_percentage = (total_return / total_invested) * 100 if total_invested > 0 else 0
        
        # Wyświetlenie statystyk
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                "💰 Wartość końcowa", 
                f"{final_value:,.0f} €",
                f"{total_return:+,.0f} €"
            )
        
        with col2:
            st.metric(
                "📊 Zwrot", 
                f"{return_percentage:+.1f}%",
                f"{'📈' if return_percentage >= 0 else '📉'}"
            )
        
        with col3:
            st.metric(
                "💵 Wpłaty regularne", 
                f"{total_regular:,.0f} €",
                f"{(total_regular/total_invested)*100:.1f}% całości"
            )
        
        with col4:
            if auto_rebalancing:
                st.metric(
                    "🤖 AUTO-REBALANCING", 
                    f"{total_rebalancing:,.0f} €",
                    f"Idealne utrzymanie proporcji"
                )
            elif band_rebalancing:
                st.metric(
                    "📊 BAND REBALANCING", 
                    f"{total_rebalancing:,.0f} €",
                    f"{len(triggers)} interwencji"
                )
            else:
                st.metric(
                    "⚖️ Budżet rebalansingu", 
                    f"{total_rebalancing:,.0f} €",
                    f"{(total_rebalancing/total_invested)*100:.1f}% całości"
                )
        
        with col5:
            actual_days = (end_date - start_date).days
            st.metric(
                "📅 Okres", 
                f"{simulation_months} miesięcy",
                f"{actual_days} dni ({actual_days/365.25:.1f} lat)"
            )
        
        # Wykresy
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📈 Rozwój wartości portfela w czasie")
            
            fig = go.Figure()
            
            # Linia łącznej wartości
            fig.add_trace(go.Scatter(
                x=df['Miesiąc'], 
                y=df['Łączna_wartość'],
                mode='lines',
                name='Łączna wartość',
                line=dict(color='orange', width=3),
                hovertemplate='Miesiąc: %{x}<br>Wartość: €%{y:,.0f}<extra></extra>'
            ))
            
            # Linie dla poszczególnych metali
            colors = {'Złoto_EUR': 'gold', 'Srebro_EUR': 'silver', 'Platyna_EUR': 'lightgray', 'Pallad_EUR': 'lightsteelblue'}
            names = {'Złoto_EUR': 'Złoto', 'Srebro_EUR': 'Srebro', 'Platyna_EUR': 'Platyna', 'Pallad_EUR': 'Pallad'}
            
            for metal, color in colors.items():
                fig.add_trace(go.Scatter(
                    x=df['Miesiąc'], 
                    y=df[metal],
                    mode='lines',
                    name=names[metal],
                    line=dict(color=color, width=2),
                    hovertemplate=f'{names[metal]}: €%{{y:,.0f}}<extra></extra>'
                ))
            
            fig.update_layout(
                xaxis_title="Miesiąc",
                yaxis_title="Wartość (EUR)",
                hovermode='x unified',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🥧 Końcowa alokacja")
            
            # Wykres kołowy
            final_values = [df['Złoto_EUR'].iloc[-1], df['Srebro_EUR'].iloc[-1], 
                          df['Platyna_EUR'].iloc[-1], df['Pallad_EUR'].iloc[-1]]
            labels = ['🥇 Złoto', '🥈 Srebro', '⚪ Platyna', '⚫ Pallad']
            colors_pie = ['gold', 'silver', 'lightgray', 'lightsteelblue']
            
            fig_pie = go.Figure(data=[go.Pie(
                labels=labels, 
                values=final_values,
                marker_colors=colors_pie,
                textinfo='label+percent',
                textposition='auto',
                hovertemplate='%{label}<br>Wartość: €%{value:,.0f}<br>Udział: %{percent}<extra></extra>'
            )])
            
            fig_pie.update_layout(height=400)
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Tabela szczegółowa portfela
        st.subheader("📋 Szczegóły końcowego portfela")
        
        final_data = df.iloc[-1]
        portfolio_details = pd.DataFrame({
            'Metal': ['🥇 Złoto', '🥈 Srebro', '⚪ Platyna', '⚫ Pallad'],
            'Gramy': [final_grams[metal] for metal in ['gold', 'silver', 'platinum', 'palladium']],
            'Cena za gram (€)': [final_prices[metal] for metal in ['gold', 'silver', 'platinum', 'palladium']],
            'Wartość (€)': [final_data['Złoto_EUR'], final_data['Srebro_EUR'], 
                           final_data['Platyna_EUR'], final_data['Pallad_EUR']],
            'Alokacja (%)': [final_data['Złoto_%'], final_data['Srebro_%'], 
                            final_data['Platyna_%'], final_data['Pallad_%']],
            'Docelowa (%)': [target_gold, target_silver, target_platinum, target_palladium]
        })
        
        portfolio_details['Różnica (%)'] = portfolio_details['Alokacja (%)'] - portfolio_details['Docelowa (%)']
        
        # Formatowanie kolumn
        portfolio_details['Gramy'] = portfolio_details['Gramy'].round(2)
        portfolio_details['Cena za gram (€)'] = portfolio_details['Cena za gram (€)'].round(3)
        portfolio_details['Wartość (€)'] = portfolio_details['Wartość (€)'].round(0)
        portfolio_details['Alokacja (%)'] = portfolio_details['Alokacja (%)'].round(1)
        portfolio_details['Różnica (%)'] = portfolio_details['Różnica (%)'].round(1)
        
        st.dataframe(portfolio_details, use_container_width=True, hide_index=True)
        
        # Wykres cen metali w czasie symulacji
        st.subheader("💹 Zmiany cen metali podczas symulacji")
        
        fig_prices = go.Figure()
        
        price_cols = [('Cena_złoto', '🥇 Złoto', 'gold'), 
                     ('Cena_srebro', '🥈 Srebro', 'silver'),
                     ('Cena_platyna', '⚪ Platyna', 'lightgray'), 
                     ('Cena_pallad', '⚫ Pallad', 'lightsteelblue')]
        
        for col, name, color in price_cols:
            fig_prices.add_trace(go.Scatter(
                x=df['Miesiąc'],
                y=df[col],
                mode='lines',
                name=name,
                line=dict(color=color, width=2),
                hovertemplate=f'{name}: €%{{y:.3f}}/g<extra></extra>'
            ))
        
        fig_prices.update_layout(
            xaxis_title="Miesiąc",
            yaxis_title="Cena (EUR/gram)",
            hovermode='x unified',
            height=300
        )
        
        st.plotly_chart(fig_prices, use_container_width=True)

elif run_simulation and total_allocation != 100:
    st.error("❌ Nie można uruchomić symulacji - suma alokacji musi wynosić 100%!")
elif run_simulation and lbma_df is None:
    st.error("❌ Nie można uruchomić symulacji - problem z wczytaniem danych LBMA!")
elif run_simulation and (start_date is None or end_date is None):
    st.error("❌ Nie można uruchomić symulacji - wybierz poprawne daty rozpoczęcia i zakończenia!")
elif run_simulation and end_date <= start_date:
    st.error("❌ Data zakończenia musi być późniejsza niż data rozpoczęcia!")

# Informacje o aplikacji
st.markdown("---")
with st.expander("ℹ️ O danych LBMA"):
    if lbma_df is not None:
        st.markdown(f"""
        **Dane wykorzystane w symulacji:**
        - 📊 **Źródło:** London Bullion Market Association (LBMA)
        - 📅 **Zakres dat:** {lbma_df['Date'].min().strftime('%Y-%m-%d')} do {lbma_df['Date'].max().strftime('%Y-%m-%d')}
        - 📈 **Liczba rekordów:** {len(lbma_df):,}
        - ⚖️ **Jednostka:** Gramy (przeliczone z uncji trojańskich)
        - 💰 **Waluta:** EUR
        
        **Przeliczenie:** 1 uncja trojańska = {TROY_OUNCE_TO_GRAMS} gramów
        """)

with st.expander("🔄 Jak działa rebalansing metodą cash-flow?"):
    st.markdown("""
    **Rebalansing metodą cash-flow** w kontekście metali szlachetnych:
    
    1. **Kupno fizycznych gramów** - każda wpłata kupuje rzeczywiste gramy metali
    2. **Brak sprzedaży** - nie sprzedajemy posiadanych gramów
    3. **Inteligentny zakup** - nowe środki kierujemy proporcjonalnie więcej do metali poniżej docelowej alokacji
    4. **Wykorzystanie zmienności cen** - automatycznie kupujemy więcej gdy ceny spadają
    5. **Minimalne koszty** - brak kosztów sprzedaży i podatków od zysków kapitałowych
    
    **Przykład:** Jeśli złoto ma stanowić 40% portfela, ale aktualnie stanowi 35%, większa część 
    nowych wpłat zostanie przeznaczona na zakup gramów złota.
    """)

# Footer
st.markdown("---")
st.markdown("*Aplikacja wykorzystuje rzeczywiste dane historyczne LBMA • Ceny w EUR za gram*")
