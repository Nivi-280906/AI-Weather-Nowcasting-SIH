"""
Seeds the database with a representative set of hyper-local monitoring
points across India's flash-flood / cloudburst-prone regions (Himalayan
foothills, Western Ghats, NE hill states, urban flood hotspots). Each
station carries a static elevation/slope/drainage-order value that a real
deployment would read from CartoDEM/SRTM.
"""
from database import SessionLocal, init_db
from models import Station

STATIONS = [
    # name, district, state, lat, lon, elevation_m, slope_deg, drainage_order
    ("Kedarnath", "Rudraprayag", "Uttarakhand", 30.7346, 79.0669, 3583, 28, 2),
    ("Rudraprayag Town", "Rudraprayag", "Uttarakhand", 30.2849, 78.9812, 895, 18, 4),
    ("Dehradun", "Dehradun", "Uttarakhand", 30.3165, 78.0322, 640, 8, 3),
    ("Leh", "Leh", "Ladakh", 34.1526, 77.5771, 3500, 12, 2),
    ("Cherrapunji", "East Khasi Hills", "Meghalaya", 25.2702, 91.7323, 1484, 22, 3),
    # Guwahati — flood-prone localities instead of one city-wide point.
    ("Guwahati - Zoo Road", "Kamrup", "Assam", 26.1583, 91.7728, 52, 2, 5),
    ("Guwahati - Anil Nagar", "Kamrup", "Assam", 26.1361, 91.7534, 54, 3, 5),
    ("Guwahati - Ganeshguri", "Kamrup", "Assam", 26.1512, 91.7815, 50, 2, 5),
    ("Munnar", "Idukki", "Kerala", 10.0889, 77.0595, 1600, 20, 3),
    ("Wayanad", "Wayanad", "Kerala", 11.6854, 76.1320, 780, 24, 2),
    # Kochi — flood-prone localities instead of one city-wide point.
    ("Kochi - Ernakulam", "Ernakulam", "Kerala", 9.9816, 76.2999, 2, 1, 5),
    ("Kochi - Kaloor", "Ernakulam", "Kerala", 9.9931, 76.2989, 3, 1, 5),
    ("Kochi - Vyttila", "Ernakulam", "Kerala", 9.9647, 76.3197, 2, 1, 5),
    ("Kochi - Edappally", "Ernakulam", "Kerala", 10.0261, 76.3080, 4, 1, 4),
    ("Mahabaleshwar", "Satara", "Maharashtra", 17.9307, 73.6477, 1353, 19, 2),
    # Mumbai — flood-prone localities instead of one city-wide point.
    ("Mumbai - Andheri", "Mumbai", "Maharashtra", 19.1197, 72.8468, 10, 2, 5),
    ("Mumbai - Kurla", "Mumbai", "Maharashtra", 19.0726, 72.8845, 8, 1, 5),
    ("Mumbai - Sion", "Mumbai", "Maharashtra", 19.0448, 72.8619, 7, 1, 5),
    ("Mumbai - Dadar (Hindmata)", "Mumbai", "Maharashtra", 19.0189, 72.8434, 6, 1, 5),
    ("Coorg (Madikeri)", "Kodagu", "Karnataka", 12.4244, 75.7382, 1525, 21, 2),
    # Bengaluru — flood-prone localities instead of one city-wide point.
    ("Bengaluru - Koramangala", "Bengaluru Urban", "Karnataka", 12.9352, 77.6245, 900, 2, 4),
    ("Bengaluru - Bellandur", "Bengaluru Urban", "Karnataka", 12.9304, 77.6784, 890, 2, 4),
    ("Bengaluru - Whitefield", "Bengaluru Urban", "Karnataka", 12.9698, 77.7500, 905, 2, 4),
    ("Bengaluru - Silk Board", "Bengaluru Urban", "Karnataka", 12.9172, 77.6228, 895, 2, 4),
    ("Shimla", "Shimla", "Himachal Pradesh", 31.1048, 77.1734, 2276, 17, 2),
    ("Kullu-Manali", "Kullu", "Himachal Pradesh", 32.2432, 77.1892, 1950, 25, 2),
    # New Delhi — flood-prone localities instead of one city-wide point.
    ("Delhi - Karol Bagh", "Central Delhi", "Delhi", 28.6517, 77.1907, 217, 1, 5),
    ("Delhi - Dwarka", "South West Delhi", "Delhi", 28.5921, 77.0460, 210, 1, 4),
    ("Delhi - Yamuna Vihar", "North East Delhi", "Delhi", 28.7089, 77.2711, 205, 1, 5),
    ("Delhi - Okhla", "South East Delhi", "Delhi", 28.5355, 77.2910, 208, 1, 5),
    # Chennai — split into flood-prone localities instead of one city-wide point,
    # so alerts resolve to a neighbourhood rather than just "Chennai".
    ("Chennai - Velachery", "Chennai", "Tamil Nadu", 12.9791, 80.2183, 4, 1, 5),
    ("Chennai - Adyar", "Chennai", "Tamil Nadu", 13.0067, 80.2570, 3, 1, 5),
    ("Chennai - T Nagar", "Chennai", "Tamil Nadu", 13.0418, 80.2341, 7, 1, 5),
    ("Chennai - Anna Nagar", "Chennai", "Tamil Nadu", 13.0850, 80.2101, 9, 1, 4),
    ("Chennai - Mylapore", "Chennai", "Tamil Nadu", 13.0339, 80.2619, 4, 1, 5),
    ("Chennai - Perambur", "Chennai", "Tamil Nadu", 13.1143, 80.2329, 10, 1, 4),
    ("Chennai - Tambaram", "Chennai", "Tamil Nadu", 12.9249, 80.1000, 22, 3, 3),
    ("Ooty", "Nilgiris", "Tamil Nadu", 11.4102, 76.6950, 2240, 23, 2),
    ("Darjeeling", "Darjeeling", "West Bengal", 27.0410, 88.2663, 2045, 26, 2),
    # Kolkata — flood-prone localities instead of one city-wide point.
    ("Kolkata - Salt Lake", "Kolkata", "West Bengal", 22.5800, 88.4180, 8, 1, 5),
    ("Kolkata - Behala", "Kolkata", "West Bengal", 22.4989, 88.3149, 6, 1, 5),
    ("Kolkata - Topsia", "Kolkata", "West Bengal", 22.5386, 88.3931, 7, 1, 5),
    ("Kolkata - Rajarhat", "Kolkata", "West Bengal", 22.5958, 88.4739, 10, 1, 4),
]


def run():
    init_db()
    db = SessionLocal()
    try:
        if db.query(Station).count() == 0:
            for name, district, state, lat, lon, elev, slope, order in STATIONS:
                db.add(Station(
                    name=name, district=district, state=state,
                    lat=lat, lon=lon, elevation_m=elev,
                    slope_deg=slope, drainage_order=order,
                ))
            db.commit()
            print(f"Seeded {len(STATIONS)} stations.")
        else:
            print("Stations already seeded, skipping.")
    finally:
        db.close()


if __name__ == "__main__":
    run()
