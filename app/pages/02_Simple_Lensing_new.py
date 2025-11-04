# This file was an intermediate/duplicate page created during debugging.
# To avoid duplicate Streamlit pages appearing, this module is intentionally
# minimal and does nothing. The canonical page is `02_Simple_Lensing.py`.

def noop():
    return None

if __name__ == '__main__':
    print('02_Simple_Lensing_new.py is a placeholder and not used.')
                
                kappa = st.session_state.get('convergence_map', None)
                if kappa is not None and hasattr(kappa, 'max'):
                    cols[0].metric("Max κ", f"{float(kappa.max()):.4f}")
                    cols[1].metric("Mean κ", f"{float(kappa.mean()):.4f}")
                    cols[2].metric("Min κ", f"{float(kappa.min()):.4f}")
                    cols[3].metric("Std κ", f"{float(kappa.std()):.4f}")
                
                # Download section
                st.markdown("### 💾 Download")
                if kappa is not None:
                    download_cols = st.columns(2)
                    
                    # Save numpy array
                    buf = io.BytesIO()
                    np.save(buf, st.session_state['convergence_map'])
                    buf.seek(0)
                    
                    with download_cols[0]:
                        st.download_button(
                            label="Download Map (.npy)",
                            data=buf,
                            file_name=f"convergence_map_{grid_size}x{grid_size}.npy",
                            mime="application/octet-stream",
                            use_container_width=True
                        )
                    
                    # Save figure
                    fig_buf = io.BytesIO()
                    fig.savefig(fig_buf, format='png', dpi=300, bbox_inches='tight')
                    fig_buf.seek(0)
                    
                    with download_cols[1]:
                        st.download_button(
                            label="Download Figure (.png)",
                            data=fig_buf,
                            file_name=f"convergence_map_{grid_size}x{grid_size}.png",
                            mime="image/png",
                            use_container_width=True
                        )
            except Exception as e:
                st.error(f"Error displaying results: {str(e)}")
                return

        # Add information expander
        with st.expander("ℹ️ About Convergence Maps"):
            st.markdown("""
            **Convergence (κ)** represents the surface mass density of a gravitational lens 
            normalized by the critical surface density. Key properties:
            
            - **κ < 1**: Weak lensing regime (small distortions)
            - **κ = 1**: Critical curve (infinite magnification)
            - **κ > 1**: Strong lensing regime (multiple images possible)
            
            **NFW Profile**: Navarro-Frenk-White profile describes the density distribution of 
            dark matter halos in cosmological simulations. It's the standard model for galaxy 
            cluster dark matter.
            
            **Elliptical NFW**: Extension that accounts for ellipticity in the projected mass 
            distribution, more realistic for observed galaxy clusters.
            """)

if __name__ == "__main__":
    main()