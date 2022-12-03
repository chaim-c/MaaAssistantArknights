#include "GeneralConfig.h"

#include "Utils/Logger.hpp"
#include <meojson/json.hpp>

bool asst::GeneralConfig::parse(const json::value& json)
{
    m_version = json.at("version").as_string();

    {
        const json::value& options_json = json.at("options");
        m_options.task_delay = options_json.at("taskDelay").as_integer();
        m_options.control_delay_lower = options_json.at("controlDelayRange")[0].as_integer();
        m_options.control_delay_upper = options_json.at("controlDelayRange")[1].as_integer();
        // m_options.print_window = options_json.at("printWindow").as_boolean();
        m_options.adb_extra_swipe_dist = options_json.get("adbExtraSwipeDist", 100);
        m_options.adb_extra_swipe_duration = options_json.get("adbExtraSwipeDuration", -1);
        m_options.adb_swipe_duration_multiplier = options_json.get("adbSwipeDurationMultiplier", 10.0);
        m_options.minitouch_extra_swipe_dist = options_json.get("minitouchExtraSwipeDist", 100);
        m_options.minitouch_extra_swipe_duration = options_json.get("minitouchExtraSwipeDuration", -1);
        if (auto order = options_json.find<json::array>("minitouchProgramsOrder")) {
            for (const auto& type : *order) {
                m_options.minitouch_programs_order.emplace_back(type.as_string());
            }
        }
        m_options.penguin_report.cmd_format = options_json.get("penguinReport", "cmdFormat", std::string());
        m_options.yituliu_report.cmd_format = options_json.get("yituliuReport", "cmdFormat", std::string());
        m_options.depot_export_template.ark_planner =
            options_json.get("depotExportTemplate", "arkPlanner", std::string());
    }

    for (const auto& [client_type, intent_name] : json.at("intent").as_object()) {
        m_intent_name[client_type] = intent_name.as_string();
    }

    for (const auto& cfg_json : json.at("connection").as_array()) {
        auto base_it = m_adb_cfg.find(cfg_json.get("baseConfig", std::string()));
        const AdbCfg& base_cfg = base_it == m_adb_cfg.end() ? AdbCfg() : base_it->second;

        AdbCfg adb;
        adb.connect = cfg_json.get("connect", base_cfg.connect);
        adb.display_id = cfg_json.get("displayId", base_cfg.display_id);
        adb.uuid = cfg_json.get("uuid", base_cfg.uuid);
        adb.click = cfg_json.get("click", base_cfg.click);
        adb.swipe = cfg_json.get("swipe", base_cfg.swipe);
        adb.press_esc = cfg_json.get("pressEsc", base_cfg.press_esc);
        adb.display = cfg_json.get("display", base_cfg.display);
        adb.screencap_raw_with_gzip = cfg_json.get("screencapRawWithGzip", base_cfg.screencap_raw_with_gzip);
        adb.screencap_raw_by_nc = cfg_json.get("screencapRawByNC", base_cfg.screencap_raw_by_nc);
        adb.nc_address = cfg_json.get("ncAddress", base_cfg.nc_address);
        adb.nc_port = static_cast<unsigned short>(cfg_json.get("ncPort", base_cfg.nc_port));
        adb.screencap_encode = cfg_json.get("screencapEncode", base_cfg.screencap_encode);
        adb.release = cfg_json.get("release", base_cfg.release);
        adb.start = cfg_json.get("start", base_cfg.start);
        adb.stop = cfg_json.get("stop", base_cfg.stop);
        adb.abilist = cfg_json.get("abilist", base_cfg.abilist);
        adb.orientation = cfg_json.get("orientation", base_cfg.orientation);
        adb.push_minitouch = cfg_json.get("pushMinitouch", base_cfg.push_minitouch);
        adb.chmod_minitouch = cfg_json.get("chmodMinitouch", base_cfg.chmod_minitouch);
        adb.call_minitouch = cfg_json.get("callMinitouch", base_cfg.call_minitouch);
        adb.call_maatouch = cfg_json.get("callMaatouch", base_cfg.call_maatouch);

        m_adb_cfg[cfg_json.at("configName").as_string()] = std::move(adb);
    }

    return true;
}