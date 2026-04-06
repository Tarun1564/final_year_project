from django.shortcuts import render
from django.utils.translation import gettext as _
from .utils import fetch_weather_data, generate_indices, aggregate_features, predict_yield, map_water_index, map_irrigation_index, map_acl_index

# Default location
DEFAULT_LAT = 17.3850
DEFAULT_LON = 78.4867

def dashboard(request):
    try:
        lat = float(request.GET.get('lat', 17.3850))
        lon = float(request.GET.get('lon', 78.4867))
        show_results = 'lat' in request.GET and 'lon' in request.GET

        daily_indices = []
        aggregated_features = []
        predicted_yield = None

        if show_results:
            weather_data = fetch_weather_data(lat, lon)
            daily_indices = generate_indices(weather_data)

            for d in daily_indices:
                d['water_msg'] = map_water_index(d['water'])
                d['irrigation_msg'] = map_irrigation_index(d['irrigation'])
                d['acl_msg'] = map_acl_index(d['acl'])

            aggregated_features = aggregate_features(daily_indices, weather_data)
            predicted_yield = predict_yield(aggregated_features)

        context = {
            "lat": lat,
            "lon": lon,
            "show_results": show_results,
            "daily_indices": daily_indices[:7],
            "aggregated_features": aggregated_features,
            "predicted_yield": predicted_yield,
        }
    except Exception as e:
        context = {"error": str(e)}

    return render(request, "index.html", context)
