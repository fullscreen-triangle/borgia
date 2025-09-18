from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns


class MolecularMLAnalyzer:
    def __init__(self, molecules_df):
        self.df = molecules_df
        self.scaler = StandardScaler()

    def pca_analysis(self):
        """Perform PCA analysis on molecular properties"""
        # Select features for PCA
        features = ['molecular_weight', 'logp', 'tpsa', 'clock_base_frequency_hz',
                    'clock_frequency_stability', 'proc_processing_rate_ops_per_sec',
                    'proc_memory_capacity_bits']

        X = self.df[features].fillna(self.df[features].mean())
        X_scaled = self.scaler.fit_transform(X)

        # Perform PCA
        pca = PCA()
        X_pca = pca.fit_transform(X_scaled)

        # Create plots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

        # Explained variance
        ax1.bar(range(1, len(pca.explained_variance_ratio_) + 1),
                pca.explained_variance_ratio_)
        ax1.set_title('PCA Explained Variance Ratio')
        ax1.set_xlabel('Principal Component')
        ax1.set_ylabel('Explained Variance Ratio')

        # Cumulative explained variance
        ax2.plot(range(1, len(pca.explained_variance_ratio_) + 1),
                 np.cumsum(pca.explained_variance_ratio_), 'bo-')
        ax2.set_title('Cumulative Explained Variance')
        ax2.set_xlabel('Principal Component')
        ax2.set_ylabel('Cumulative Explained Variance')
        ax2.grid(True, alpha=0.3)

        # PCA scatter plot
        scatter = ax3.scatter(X_pca[:, 0], X_pca[:, 1],
                              c=self.df['molecular_weight'], cmap='viridis', alpha=0.7)
        ax3.set_title('PCA: First Two Components')
        ax3.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        ax3.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.colorbar(scatter, ax=ax3, label='Molecular Weight')

        plt.tight_layout()
        return fig, pca

    def clustering_analysis(self, n_clusters=4):
        """Perform K-means clustering on molecular data"""
        features = ['molecular_weight', 'logp', 'tpsa', 'clock_base_frequency_hz',
                    'clock_frequency_stability', 'proc_processing_rate_ops_per_sec']

        X = self.df[features].fillna(self.df[features].mean())
        X_scaled = self.scaler.fit_transform(X)

        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)

        # Add clusters to dataframe
        self.df['cluster'] = clusters

        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'K-Means Clustering Analysis (k={n_clusters})', fontsize=16)

        # Cluster scatter plots
        scatter1 = axes[0, 0].scatter(self.df['molecular_weight'], self.df['logp'],
                                      c=clusters, cmap='tab10', alpha=0.7)
        axes[0, 0].set_title('Clusters: Molecular Weight vs LogP')
        axes[0, 0].set_xlabel('Molecular Weight')
        axes[0, 0].set_ylabel('LogP')

        scatter2 = axes[0, 1].scatter(self.df['clock_base_frequency_hz'] / 1e12,
                                      self.df['proc_processing_rate_ops_per_sec'] / 1e6,
                                      c=clusters, cmap='tab10', alpha=0.7)
        axes[0, 1].set_title('Clusters: Clock Frequency vs Processing Rate')
        axes[0, 1].set_xlabel('Clock Frequency (THz)')
        axes[0, 1].set_ylabel('Processing Rate (Mops/sec)')

        # Cluster characteristics
        cluster_stats = self.df.groupby('cluster')[features].mean()

        # Heatmap of cluster characteristics
        sns.heatmap(cluster_stats.T, annot=True, cmap='RdYlBu_r', ax=axes[1, 0])
        axes[1, 0].set_title('Cluster Characteristics (Mean Values)')
        axes[1, 0].set_ylabel('Features')

        # Cluster sizes
        cluster_counts = pd.Series(clusters).value_counts().sort_index()
        axes[1, 1].bar(cluster_counts.index, cluster_counts.values,
                       color=plt.cm.tab10(np.arange(n_clusters)))
        axes[1, 1].set_title('Cluster Sizes')
        axes[1, 1].set_xlabel('Cluster')
        axes[1, 1].set_ylabel('Number of Molecules')

        plt.tight_layout()
        return fig, kmeans

    def feature_importance_analysis(self):
        """Analyze feature importance for predicting clock frequency"""
        features = ['molecular_weight', 'logp', 'tpsa', 'clock_frequency_stability',
                    'proc_processing_rate_ops_per_sec', 'proc_memory_capacity_bits',
                    'proc_parallel_processing_num']

        X = self.df[features].fillna(self.df[features].mean())
        y = self.df['clock_base_frequency_hz']

        # Train Random Forest
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)

        # Get feature importance
        importance = rf.feature_importances_
        feature_names = features

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        indices = np.argsort(importance)[::-1]

        ax.bar(range(len(features)), importance[indices])
        ax.set_title('Feature Importance for Predicting Clock Frequency')
        ax.set_xlabel('Features')
        ax.set_ylabel('Importance')
        ax.set_xticks(range(len(features)))
        ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')

        plt.tight_layout()
        return fig, rf


# Usage example
def run_ml_analysis(molecules_df):
    ml_analyzer = MolecularMLAnalyzer(molecules_df)

    # PCA Analysis
    pca_fig, pca_model = ml_analyzer.pca_analysis()
    pca_fig.savefig('pca_analysis.png', dpi=300, bbox_inches='tight')

    # Clustering Analysis
    cluster_fig, kmeans_model = ml_analyzer.clustering_analysis()
    cluster_fig.savefig('clustering_analysis.png', dpi=300, bbox_inches='tight')

    # Feature Importance
    importance_fig, rf_model = ml_analyzer.feature_importance_analysis()
    importance_fig.savefig('feature_importance.png', dpi=300, bbox_inches='tight')

    return pca_fig, cluster_fig, importance_fig
