#pragma once
#include "Task/InterfaceTask.h"

#include <memory>

namespace asst
{
    class BattleProcessTask;
    class BattleFormationTask;
    class ProcessTask;

    // 保全派驻抄作业任务
    class SSSCopilotTask final : public InterfaceTask
    {
    public:
        inline static constexpr std::string_view TaskType = "SSSCopilot";

        SSSCopilotTask(const AsstCallback& callback, Assistant* inst);
        virtual ~SSSCopilotTask() override = default;

        virtual bool set_params(const json::value& params) override;

    private:
        std::shared_ptr<ProcessTask> m_begin_task_ptr = nullptr;
        std::shared_ptr<BattleFormationTask> m_formation_task_ptr = nullptr;
        std::shared_ptr<BattleProcessTask> m_battle_task_ptr = nullptr;
    };
}
