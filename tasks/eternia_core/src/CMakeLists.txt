# ------------------------------------------------------------------------------
# Set variables
# ------------------------------------------------------------------------------
set(MOD_EXPORTS ${REPO_NAMESPACE}_eternia_core_exports)

# ------------------------------------------------------------------------------
# Build eternia_core Task Library
# ------------------------------------------------------------------------------
add_chimod_runtime_lib(${REPO_NAMESPACE} eternia_core eternia_core_runtime.cc)
add_chimod_client_lib(${REPO_NAMESPACE} eternia_core eternia_core_client.cc)

target_link_libraries(eternia_eternia_core_client PUBLIC hermes::hermes_core_client)
target_link_libraries(eternia_eternia_core_client_gpu PUBLIC hermes::hermes_core_client_gpu)
target_link_libraries(eternia_eternia_core PUBLIC hermes::hermes_core_client_gpu)

# ------------------------------------------------------------------------------
# Install eternia_core Task Library
# ------------------------------------------------------------------------------
install(
        TARGETS
        ${${MOD_EXPORTS}}
        EXPORT
        ${CHIMAERA_EXPORTED_TARGETS}
        LIBRARY DESTINATION ${CHIMAERA_INSTALL_LIB_DIR}
        ARCHIVE DESTINATION ${CHIMAERA_INSTALL_LIB_DIR}
        RUNTIME DESTINATION ${CHIMAERA_INSTALL_BIN_DIR}
)
install(FILES eternia_core_monitor.py
        DESTINATION ${CHIMAERA_INSTALL_LIB_DIR})

# ------------------------------------------------------------------------------
# Coverage
# ------------------------------------------------------------------------------
if(CHIMAERA_ENABLE_COVERAGE)
        set_coverage_flags(eternia_core)
endif()
