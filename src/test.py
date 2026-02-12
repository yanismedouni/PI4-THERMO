"""
Test simple pour vérifier que toutes les dépendances fonctionnent
"""

print("="*70)
print("TEST DE L'ENVIRONNEMENT")
print("="*70)

# ------------------------------------------------------------------
# Test 1 : Python
# ------------------------------------------------------------------
print("\n1. Version Python:")
import sys
print(f"   ✓ Python {sys.version}")

# ------------------------------------------------------------------
# Test 2 : NumPy
# ------------------------------------------------------------------
print("\n2. NumPy:")
try:
    import numpy as np
    print(f"   ✓ NumPy version {np.__version__}")
    arr = np.array([1, 2, 3])
    print(f"   ✓ Test sum([1,2,3]) = {np.sum(arr)}")
except ImportError as e:
    print(f"   ✗ ERREUR: {e}")

# ------------------------------------------------------------------
# Test 3 : Pandas
# ------------------------------------------------------------------
print("\n3. Pandas:")
try:
    import pandas as pd
    print(f"   ✓ Pandas version {pd.__version__}")
    df_test = pd.DataFrame({"a":[1,2], "b":[3,4]})
    print(f"   ✓ DataFrame test OK")
except ImportError as e:
    print(f"   ✗ ERREUR: {e}")

# ------------------------------------------------------------------
# Test 4 : Scikit-learn
# ------------------------------------------------------------------
print("\n4. Scikit-learn:")
try:
    import sklearn
    from sklearn.preprocessing import RobustScaler
    from sklearn.cluster import DBSCAN
    
    print(f"   ✓ sklearn version {sklearn.__version__}")
    
    # Test rapide scaler + DBSCAN
    X = np.array([[1,2],[2,3],[10,10]])
    scaler = RobustScaler()
    Xs = scaler.fit_transform(X)
    
    model = DBSCAN(eps=1.5, min_samples=2)
    labels = model.fit_predict(Xs)
    
    print(f"   ✓ RobustScaler + DBSCAN OK")
except ImportError as e:
    print(f"   ✗ ERREUR: {e}")

# ------------------------------------------------------------------
# Test 5 : Matplotlib
# ------------------------------------------------------------------
print("\n5. Matplotlib:")
try:
    import matplotlib
    print(f"   ✓ Matplotlib version {matplotlib.__version__}")
except ImportError as e:
    print(f"   ✗ ERREUR: {e}")

# ------------------------------------------------------------------
# Test 6 : SciPy
# ------------------------------------------------------------------
print("\n6. SciPy:")
try:
    import scipy
    print(f"   ✓ SciPy version {scipy.__version__}")
except ImportError as e:
    print(f"   ✗ ERREUR: {e}")

# ------------------------------------------------------------------
# Test 8 : MOSEK
# ------------------------------------------------------------------
print("\n8. MOSEK:")
try:
    import mosek
    print(f"   ✓ MOSEK importé avec succès")
    
    from mosek.fusion import Model, Domain
    with Model("test") as M:
        x = M.variable("x", 1, Domain.greaterThan(0.0))
        print(f"   ✓ Modèle MOSEK créé avec succès")
    
except ImportError as e:
    print(f"   ✗ ERREUR d'import: {e}")
except Exception as e:
    print(f"   ✗ ERREUR MOSEK: {e}")
    print("   → Vérifiez votre licence MOSEK")

# ------------------------------------------------------------------
# Test 9 : Calcul math simple
# ------------------------------------------------------------------
print("\n9. Test calcul:")
try:
    result = np.sin(np.pi/2)
    print(f"   ✓ sin(π/2) = {result:.4f}")
except Exception as e:
    print(f"   ✗ ERREUR: {e}")

print("\n" + "="*70)
print("FIN DES TESTS")
print("="*70)
