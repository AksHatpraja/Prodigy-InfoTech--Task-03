import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from skimage.feature import hog

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

BASE_DIR       = r"C:\Users\MY PC\OneDrive\Desktop\PRODIGY_ML_03-main"
DATA_DIR       = os.path.join(BASE_DIR, "data", "petImages")
IMG_SIZE       = 128
MAX_PER_CLASS  = 2000
PCA_COMPONENTS = 350
SVM_KERNEL     = "rbf"
SVM_C          = 1
RANDOM_STATE   = 42


def check_folders():
    print("=" * 55)
    print("   SVM Cats vs Dogs Classifier  (HOG + PCA + SVM)")
    print("=" * 55)

    cat_dir = os.path.join(DATA_DIR, "Cat")
    dog_dir = os.path.join(DATA_DIR, "Dog")
    errors  = []

    if not os.path.exists(DATA_DIR):
        errors.append(f"  [X] Not found: {DATA_DIR}")
    else:
        print(f"  [OK] petImages folder found: {DATA_DIR}")

    if not os.path.exists(cat_dir):
        errors.append(f"  [X] Not found: {cat_dir}")
    else:
        cat_files = [f for f in os.listdir(cat_dir)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"  [OK] Cat folder found — {len(cat_files)} images")

    if not os.path.exists(dog_dir):
        errors.append(f"  [X] Not found: {dog_dir}")
    else:
        dog_files = [f for f in os.listdir(dog_dir)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"  [OK] Dog folder found — {len(dog_files)} images")

    if errors:
        print("\n  ERROR — Following folders not found:")
        for e in errors:
            print(e)
        print("\n  Expected structure:")
        print(r"  C:\Users\MY PC\OneDrive\Desktop\PRODIGY_ML_03-main\data\PetImages\Cat")
        print(r"  C:\Users\MY PC\OneDrive\Desktop\PRODIGY_ML_03-main\data\PetImages\Dog")
        exit(1)

    print()
    return cat_dir, dog_dir


def extract_hog_features(img):
    features = hog(
        img,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        visualize=False,
    )
    return features


def load_images(cat_dir, dog_dir, img_size, max_per_class):
    X_raw  = []
    X_hog  = []
    y      = []

    class_info = [
        (cat_dir, 0, "Cat", max_per_class),
        (dog_dir, 1, "Dog", max_per_class),
    ]

    for folder, label, class_name, limit in class_info:
        count = 0
        files = os.listdir(folder)
        print(f"Loading {class_name} images...")

        for fname in tqdm(files, desc=f"  {class_name}"):
            if count >= limit:
                break
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                continue

            path = os.path.join(folder, fname)
            img  = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            img = cv2.resize(img, (img_size, img_size))

            # Raw pixels (normalized)
            raw = img.flatten() / 255.0

            # HOG features
            hog_feat = extract_hog_features(img)

            X_raw.append(raw)
            X_hog.append(hog_feat)
            y.append(label)
            count += 1

        print(f"  Loaded: {count} {class_name} images\n")

    X_raw = np.array(X_raw)
    X_hog = np.array(X_hog)
    y     = np.array(y)

    # Combine raw pixels + HOG features
    X_combined = np.hstack([X_raw, X_hog])

    print(f"Raw pixel features  : {X_raw.shape[1]}")
    print(f"HOG features        : {X_hog.shape[1]}")
    print(f"Combined features   : {X_combined.shape[1]}")
    print(f"Total images        : {X_combined.shape[0]}")
    print(f"Cats (0)            : {(y == 0).sum()}")
    print(f"Dogs (1)            : {(y == 1).sum()}")
    print()

    return X_combined, y


def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"Train samples : {len(X_train)}")
    print(f"Test samples  : {len(X_test)}")
    print()
    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test):
    scaler      = StandardScaler()
    X_train_sc  = scaler.fit_transform(X_train)
    X_test_sc   = scaler.transform(X_test)
    print("Feature scaling done (StandardScaler)")
    print()
    return scaler, X_train_sc, X_test_sc


def apply_pca(X_train, X_test, n_components):
    print(f"PCA: {X_train.shape[1]} features -> {n_components} components")

    pca = PCA(
        n_components=n_components,
        whiten=True,
        random_state=RANDOM_STATE,
    )

    X_train_pca = pca.fit_transform(X_train)
    X_test_pca  = pca.transform(X_test)

    explained = pca.explained_variance_ratio_.sum()
    print(f"Variance retained : {explained:.1%}")
    print()
    return pca, X_train_pca, X_test_pca


def train_svm(X_train_pca, y_train):
    print(f"Training SVM (kernel='{SVM_KERNEL}', C={SVM_C}) ...")
    print("Please wait — this may take 5 to 15 minutes ...\n")

    svm = SVC(
        kernel=SVM_KERNEL,
        C=SVM_C,
        gamma="scale",
        random_state=RANDOM_STATE,
    )
    svm.fit(X_train_pca, y_train)

    print(f"Training complete. Support vectors: {svm.n_support_.sum()}")
    print()
    return svm


def evaluate(svm, X_test_pca, y_test):
    y_pred = svm.predict(X_test_pca)
    acc    = accuracy_score(y_test, y_pred)

    print("=" * 55)
    print(f"  ACCURACY : {acc:.2%}")
    print("=" * 55)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Cat", "Dog"]))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Predicted Cat", "Predicted Dog"],
        yticklabels=["Actual Cat",    "Actual Dog"],
    )
    plt.title("Confusion Matrix", fontsize=14, pad=12)
    plt.tight_layout()

    save_path = os.path.join(BASE_DIR, "confusion_matrix.png")
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Saved: {save_path}\n")

    return y_pred, acc


def visualize_predictions(X_test, y_test, y_pred, n=10):
    label_names = {0: "Cat", 1: "Dog"}
    fig, axes   = plt.subplots(2, 5, figsize=(16, 7))
    axes        = axes.flatten()

    for i in range(n):
        # Show only raw pixel part (first IMG_SIZE*IMG_SIZE values)
        img      = X_test[i][:IMG_SIZE * IMG_SIZE].reshape(IMG_SIZE, IMG_SIZE)
        true_lbl = label_names[y_test[i]]
        pred_lbl = label_names[y_pred[i]]
        color    = "green" if y_test[i] == y_pred[i] else "red"

        axes[i].imshow(img, cmap="gray")
        axes[i].set_title(
            f"True: {true_lbl}\nPred: {pred_lbl}",
            color=color, fontsize=11
        )
        axes[i].axis("off")

    plt.suptitle(
        "Sample Predictions  (Green = Correct | Red = Wrong)",
        fontsize=13
    )
    plt.tight_layout()

    save_path = os.path.join(BASE_DIR, "sample_predictions.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {save_path}\n")


def plot_pca_variance(pca):
    cumulative = np.cumsum(pca.explained_variance_ratio_)

    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(cumulative) + 1), cumulative, linewidth=2, color="steelblue")
    plt.axhline(y=0.95, color="red", linestyle="--", label="95% threshold")
    plt.xlabel("Number of PCA Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("PCA Explained Variance")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    save_path = os.path.join(BASE_DIR, "pca_variance.png")
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Saved: {save_path}\n")


def main():
    # Step 1: Check folders
    cat_dir, dog_dir = check_folders()

    # Step 2: Load images + extract HOG features
    X, y = load_images(cat_dir, dog_dir, IMG_SIZE, MAX_PER_CLASS)

    # Step 3: Train/Test split
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Step 4: Scale features
    scaler, X_train_sc, X_test_sc = scale_features(X_train, X_test)

    # Step 5: PCA
    pca, X_train_pca, X_test_pca = apply_pca(X_train_sc, X_test_sc, PCA_COMPONENTS)

    # Step 6: PCA variance plot
    plot_pca_variance(pca)

    # Step 7: Train SVM
    svm = train_svm(X_train_pca, y_train)

    # Step 8: Evaluate
    y_pred, accuracy = evaluate(svm, X_test_pca, y_test)

    # Step 9: Visualize predictions
    visualize_predictions(X_test, y_test, y_pred, n=10)

    print("=" * 55)
    print(f"  DONE! Final Accuracy: {accuracy:.2%}")
    print(f"  Output files saved in: {BASE_DIR}")
    print("  1. confusion_matrix.png")
    print("  2. pca_variance.png")
    print("  3. sample_predictions.png")
    print("=" * 55)


if __name__ == "__main__":
    main()
