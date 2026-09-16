import { Stepper } from '@mantine/core'
import { PageHeader } from './PageHeader'
import './setup.css'

const steps = ['创建项目', '导入记录', '确认人物', '建立图谱']

/** 只标记当前位置，不把前面的步骤推断为完成，适用于追加导入。 */
export function SetupHeader({ step, title, description }: { step: number; title: string; description: string }) {
  return <header className="setup-header">
    <Stepper className="setup-steps" active={step} size="sm" color="moon" aria-label="资料准备步骤">
      {steps.map((label, index) => (
        <Stepper.Step
          key={label}
          label={label}
          icon={index + 1}
          completedIcon={index + 1}
          aria-current={index === step ? 'step' : undefined}
        />
      ))}
    </Stepper>
    <PageHeader title={title} description={description} />
  </header>
}
